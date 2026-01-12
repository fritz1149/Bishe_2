import json
import time
import os
import sys
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

# DeepSpeed 初始化需尽早导入
import deepspeed

# 删除原有的 init_distributed，DeepSpeed 会处理分布式
# 保留 is_distributed 等辅助函数用于日志

def resume(model, args):
    if getattr(args, "resume_encoder", None) and args.resume_encoder != "":
        ckpt = torch.load(args.resume_encoder, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        encoder_state_dict = {k[len("module.backbone.original_model."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.original_model.")}
        incompatible_keys = model.encoder.original_model.load_state_dict(encoder_state_dict, strict=False)
        if args.resume_log:
            print("resume encoder")
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)
    if getattr(args, "resume_linear", None) and args.resume_linear != "":
        ckpt = torch.load(args.resume_linear, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        linear_state_dict = {k[len("encoder.fc."):]: v for k, v in state_dict.items() if k.startswith("encoder.fc.")}
        incompatible_keys = model.encoder.fc.load_state_dict(linear_state_dict, strict=False)
        if args.resume_log:
            print("resume linear")
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)
    if getattr(args, "resume_lora0", None) and args.resume_lora0 != "":
        ckpt = torch.load(args.resume_lora0, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        lora0_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
        incompatible_keys = model.backbone.base_model.model.model.language_model.layers.load_state_dict(lora0_state_dict, strict=False)
        if args.resume_log:
            print("resume lora0")
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)
    if getattr(args, "resume_lora1", None) and args.resume_lora1 != "":
        ckpt = torch.load(args.resume_lora1, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        lora1_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
        incompatible_keys = model.backbone.base_model.model.model.language_model.layers.load_state_dict(lora1_state_dict, strict=False)
        if args.resume_log:
            print("resume lora1")
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self): 
        return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'.format(**self.__dict__)

def save_checkpoint(state, is_best, args, filename="checkpoint.pt"):
    """保存轻量级元数据 checkpoint（用于兼容性）"""
    path = os.path.join(args.save_dir, filename)
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_checkpoint.pt")
    torch.save(state, path)
    if is_best:
        if os.path.exists(best_path):
            os.remove(best_path)
        os.link(path, best_path)

def save_deepspeed_checkpoint(model_engine, epoch, best_loss, best_acc, args, tag=None):
    """使用 DeepSpeed API 保存完整 checkpoint（包括 optimizer 和 scheduler）"""
    os.makedirs(args.save_dir, exist_ok=True)
    if tag is None:
        tag = f"epoch_{epoch}"
    checkpoint_dir = os.path.join(args.save_dir, tag)
    # 使用 client_state 保存训练元数据（DeepSpeed 推荐方式）
    client_state = dict(epoch=epoch, best_loss=best_loss, best_acc=best_acc)
    # DeepSpeed 会自动保存模型、optimizer、scheduler 状态
    model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)
    if args.is_master:
        print(f"Saved DeepSpeed checkpoint to {checkpoint_dir} (epoch={epoch}, best_loss={best_loss:.4f}, best_acc={best_acc:.4f})")

def load_deepspeed_checkpoint(model_engine, checkpoint_dir, args):
    """使用 DeepSpeed API 加载完整 checkpoint（包括 optimizer 和 scheduler）"""
    if not os.path.exists(checkpoint_dir):
        if args.is_master:
            print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return None, 0, float("inf"), float("-inf")
    
    # DeepSpeed 会自动加载模型、optimizer、scheduler 状态
    # load_optimizer_states=True 和 load_lr_scheduler_states=True 确保恢复 optimizer 和 scheduler
    _, client_state = model_engine.load_checkpoint(
        checkpoint_dir, 
        load_optimizer_states=True, 
        load_lr_scheduler_states=True
    )
    
    # 从 client_state 获取训练元数据
    if client_state:
        epoch = client_state.get("epoch", 0)
        best_loss = client_state.get("best_loss", float("inf"))
        best_acc = client_state.get("best_acc", float("-inf"))
    else:
        # 如果没有 client_state，尝试从旧格式的元数据文件加载（向后兼容）
        metadata_path = os.path.join(checkpoint_dir, "training_metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
            epoch = metadata.get("epoch", 0)
            best_loss = metadata.get("best_loss", float("inf"))
            best_acc = metadata.get("best_acc", float("-inf"))
        else:
            epoch = 0
            best_loss = float("inf")
            best_acc = float("-inf")
    
    if args.is_master:
        print(f"Loaded DeepSpeed checkpoint from {checkpoint_dir}")
        print(f"Resuming from epoch {epoch}, best_loss={best_loss:.4f}, best_acc={best_acc:.4f}")
    
    return client_state, epoch, best_loss, best_acc

def train(args):
    # DeepSpeed 会自动初始化分布式
    if args.is_master:
        for k, v in vars(args).items():
            print(f"{k}: {v}")

    from model import ProposeModel
    model = ProposeModel(args)

    # ✅ 关键：不再手动分片到 cuda:0/cuda:1，让 DeepSpeed 控制
    # model.encoder = model.encoder.to('cuda:0')  # ← 删除
    # model.text_embedder = model.text_embedder.to('cuda:0')  # ← 删除

    from custom_datasets import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.train_dir)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_LLMDataset
    )

    # ✅ 使用 DeepSpeed 初始化
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters_(),  # 仅 LoRA 参数
        config=args.deepspeed_config
    )

    # 加载预训练权重（如果指定）
    resume(model, args)
    
    # 尝试从 DeepSpeed checkpoint 恢复训练（如果指定）
    start_epoch = 0
    best_loss = float("inf")
    best_acc = float("-inf")
    if getattr(args, "resume_from_checkpoint", None) and args.resume_from_checkpoint != "":
        _, start_epoch, best_loss, best_acc = load_deepspeed_checkpoint(
            model_engine, args.resume_from_checkpoint, args
        )
        # start_epoch 应该从下一个 epoch 开始
        start_epoch = start_epoch + 1
    
    last_logging = time.time()

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter("Loss", ":.4f")
        model_engine.train()
        model_engine.gradient_checkpointing_enable()
        model_engine.enable_input_require_grads()

        for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels, _) in enumerate(loader, start=epoch * len(loader)):
            assert input_ids.shape[1] <= 4096
            result: Qwen3VLCausalLMOutputWithPast = model_engine(
                input_ids=input_ids,
                labels=labels_ids,
                payloads=payloads,
                position_ids=position_ids,
                attention_mask=attention_mask,
                rope_deltas=None
            )
            loss = result.loss / args.accumulation_steps
            model_engine.backward(loss)
            losses.update(loss.item(), len(labels))

            if (step + 1) % args.accumulation_steps == 0:
                model_engine.step()  # 自动 optimizer.step + scheduler.step + zero_grad

            if args.is_master and step % args.log_freq == 0:
                current_time = time.time()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {losses.avg:.4f}, Time: {current_time - last_logging:.2f}s")
                last_logging = current_time
                sys.stdout.flush()

        # Eval
        eval_loss, eval_acc, pre, rec, f1 = eval(args, model_engine.module if hasattr(model_engine, 'module') else model_engine)
        if args.is_master:
            is_best = eval_acc > best_acc or (eval_acc == best_acc and losses.avg < best_loss)
            if is_best: 
                best_loss = losses.avg
                best_acc = eval_acc
            if (epoch % args.save_freq == 0) or is_best:
                # 保存 DeepSpeed checkpoint（包括 optimizer 和 scheduler）
                tag = f"epoch_{epoch}"
                if is_best:
                    tag = "best"
                save_deepspeed_checkpoint(model_engine, epoch, best_loss, best_acc, args, tag=tag)
                
                # 同时保存轻量级 checkpoint（用于兼容性，仅包含模型权重）
                state = dict(
                    epoch=epoch, 
                    model=model.state_dict_(),  # 调用你已有的 state_dict_ 方法
                    best_loss=best_loss, 
                    best_acc=best_acc
                )
                save_checkpoint(state, is_best, args)
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=best_loss, best_acc=best_acc, eval_acc=eval_acc)
            print(json.dumps(logs))
            print(f"Epoch {epoch} => Eval accuracy ({eval_acc:.4f} %)")

    return

def eval(args, model=None):
    if model is None:
        from model import ProposeModel
        model = ProposeModel(args)
        resume(model, args)
    
    model.eval()
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()

    from custom_datasets import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.test_dir)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.per_device_batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_LLMDataset
    )

    loss_list = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels, _) in enumerate(loader):
            assert input_ids.shape[1] <= 4096
            result: Qwen3VLCausalLMOutputWithPast = model(
                input_ids=input_ids,
                labels=labels_ids,
                payloads=payloads,
                position_ids=position_ids,
                attention_mask=attention_mask,
                rope_deltas=None
            )
            loss = result.loss
            loss_list.append(loss.item())
    if len(loss_list) > 0:
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Eval Loss: avg={avg_loss:.4f} (samples={len(loss_list)})")
    else:
        print("No valid samples for eval loss.")
    # avg_loss = 0.0

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_LLMDataset
    )

    correct_count = total_count = tp = fp = fn = 0
    for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels, rope_deltas) in enumerate(loader):
        assert input_ids.shape[1] <= 4096
        first_not_minus_100 = (labels_ids[0] != -100).nonzero()[0].item()
        input_ids_ = input_ids[:, :first_not_minus_100]
        position_ids = position_ids[:, :, :first_not_minus_100]
        attention_mask = attention_mask[:, :first_not_minus_100]
        result = model.generate(
            input_ids=input_ids_,
            attention_mask=attention_mask,
            position_ids=position_ids,
            payloads=payloads,
            rope_deltas=rope_deltas,
            max_new_tokens=32,
            do_sample=False
        )
        result = result.to('cpu')
        from preprocess.utils import _ids_to_str
        print("input_label_part:", _ids_to_str(input_ids[0][first_not_minus_100:], type="qwen3vl"))
        print("ouput_label_part:", _ids_to_str(result[0][first_not_minus_100:input_ids.shape[1]], type="qwen3vl"))
        print("==========================================================")
        sys.stdout.flush()
        is_same = torch.equal(result[0][first_not_minus_100:input_ids.shape[1]], input_ids[0][first_not_minus_100:])
        # 统计
        total_count += 1
        if is_same:
            correct_count += 1
            tp += 1
        else:
            fn += 1

    if total_count > 0:
        acc = correct_count / total_count
    else:
        acc = 0.0
    
    if tp + fp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0.0
    
    if tp + fn > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0.0
    
    if pre + rec > 0:
        f1 = 2 * (pre * rec) / (pre + rec)
    else:
        f1 = 0.0

    print(f"Evaluation Metrics:")
    print(f"  Total samples: {total_count}")
    print(f"  Correct samples: {correct_count}")
    print(f"  Accuracy (ACC): {acc:.4f}")
    print(f"  Precision (PRE): {pre:.4f}")
    print(f"  Recall (REC): {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    return avg_loss, acc, pre, rec, f1

def add_args(parser):
    parser.add_argument('--finetune_mode', action='store_true', default=False)
    parser.add_argument('--align1_mode', action='store_true', default=False)
    parser.add_argument('--align2_mode', action='store_true', default=False)
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/datasets/train/')
    parser.add_argument('--test_dir', type=str, default='/datasets/test/')
    parser.add_argument('--save_dir', type=str, default='./out/')
    parser.add_argument('--log_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume_encoder', type=str, default="")
    parser.add_argument('--resume_linear', type=str, default="")
    parser.add_argument('--resume_lora0', type=str, default="")
    parser.add_argument('--resume_lora1', type=str, default="")
    parser.add_argument('--resume_log', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default="", 
                        help='DeepSpeed checkpoint 目录路径，用于恢复训练（包括 optimizer 和 scheduler）')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)  # total batch size
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--projector', type=str, default='linear')
    parser.add_argument('--linear_output_dim', type=int, default=4096)
    parser.add_argument('--llm', type=str, default='Qwen3VL-3B-Instruct')

    parser.add_argument('--deepspeed_config', type=str, default='./deepspeed_lora_config.json', help='DeepSpeed 配置文件路径')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()

    # DeepSpeed 需要设置 is_master（通过 LOCAL_RANK）
    args.is_master = (int(os.environ.get('LOCAL_RANK', 0)) == 0)

    if args.eval_mode:
        eval(args, None)
    elif args.finetune_mode or args.align1_mode or args.align2_mode:
        train(args)
    else:
        raise ValueError("Invalid mode!")