import json
import time
from torch import nn
from torch import distributed as dist
import os
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from accelerate import load_checkpoint_and_dispatch, dispatch_model

def is_distributed(): return False if not (dist.is_available and dist.is_initialized()) else True
def get_rank(): return dist.get_rank() if is_distributed () else 0
def get_world_size(): return dist.get_world_size() if is_distributed else 0
def init_distributed(args):
    ddp = int(os.environ.get('RANK', -1)) != -1

    if not (ddp and args.distributed):
        args.rank, args.world_size, args.is_master, args.gpu = 0, 1, True, None
        args.device = torch.device(args.device)
        args.per_device_batch_size = args.batch_size
        print("Not using distributed mode!")
    else:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.is_master = args.rank == 0
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        print(f'| distributed init (rank {args.rank}), gpu {args.gpu}', flush=True)
        dist.init_process_group(backend="nccl", world_size=args.world_size, rank=args.rank, device_id=args.device)

        # update config
        args.per_device_batch_size = int(args.batch_size / torch.cuda.device_count())
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)

    return is_distributed()

class AverageMeter:
    """Computes and stores the average and current value"""
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
    
    def __str__(self): return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'.format(**self.__dict__)

def resume(model, args):
    if getattr(args, "resume_encoder", None) and args.resume_encoder != "":
        ckpt = torch.load(args.resume_encoder, map_location="cpu", weights_only=True)
        # 仅加载 "encoder." 开头的权重
        state_dict = ckpt["model"]
        encoder_state_dict = {k[len("module.backbone.original_model."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.original_model.")}
        incompatible_keys = model.encoder.original_model.load_state_dict(encoder_state_dict, strict=False)
        if args.resume_log:
            print("resume encoder")
            for k in state_dict.keys():
                print(k)
            print("Missing keys (模型有，但 checkpoint 没有):")
            print(incompatible_keys.missing_keys)
            print("\nUnexpected keys (checkpoint 有，但模型没有):")
            print(incompatible_keys.unexpected_keys)
    if getattr(args, "resume_linear", None) and args.resume_linear != "":
        ckpt = torch.load(args.resume_linear, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        linear_state_dict = {k[len("encoder.fc."):]: v for k, v in state_dict.items() if k.startswith("encoder.fc.")}
        incompatible_keys = model.encoder.fc.load_state_dict(linear_state_dict, strict=False)
        if args.resume_log:
            print("resume linear")
            for k in state_dict.keys():
                print(k)
            print("Missing keys (模型有，但 checkpoint 没有):")
            print(incompatible_keys.missing_keys)
            print("\nUnexpected keys (checkpoint 有，但模型没有):")
            print(incompatible_keys.unexpected_keys)
    if getattr(args, "resume_lora0", None) and args.resume_lora0 != "":
        ckpt = torch.load(args.resume_lora0, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        lora0_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
        incompatible_keys = model.backbone.base_model.model.model.language_model.layers.load_state_dict(lora0_state_dict, strict=False)
        if args.resume_log:
            print("resume lora0")
            for k in state_dict.keys():
                print(k)
            print("Missing keys (模型有，但 checkpoint 没有):")
            print(incompatible_keys.missing_keys)
            print("\nUnexpected keys (checkpoint 有，但模型没有):")
            print(incompatible_keys.unexpected_keys)
    if getattr(args, "resume_lora1", None) and args.resume_lora1 != "":
        ckpt = torch.load(args.resume_lora1, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        lora1_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
        incompatible_keys = model.backbone.base_model.model.model.language_model.layers.load_state_dict(lora1_state_dict, strict=False)
        if args.resume_log:
            print("resume lora1")
            for k in state_dict.keys():
                print(k)
            print("Missing keys (模型有，但 checkpoint 没有):")
            print(incompatible_keys.missing_keys)
            print("\nUnexpected keys (checkpoint 有，但模型没有):")
            print(incompatible_keys.unexpected_keys)
def save_checkpoint(state, is_best, args, filename="checkpoint.pt"):
    # import time
    # start_time = time.time()
    path = os.path.join(args.save_dir, filename)
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_checkpoint.pt")
    
    torch.save(state, path)
    
    # 如果是最佳模型，需要保存到 best_checkpoint.pt
    if is_best:
        if os.path.exists(best_path):
            os.remove(best_path)
        os.link(path, best_path)
    
    # end_time = time.time()
    # print(f"保存 checkpoint 用时: {end_time - start_time:.4f} 秒")
    # sys.stdout.flush()
    return


def train(args):
    # init
    init_distributed(args)
    if args.is_master:
        for k, v in vars(args).items():
            print(f"{k}: {v}")
            
    start_epoch = 0
    best_loss = float("inf")
    from model import ProposeModel
    model = ProposeModel(args)
    # print(model)
    # return

    model.encoder = model.encoder.to('cuda:0')
    model.text_embedder = model.text_embedder.to('cuda:0')
    device_map = {
        "base_model.model.model.language_model.embed_tokens": "cuda:0",
        "base_model.model.model.language_model.rotary_emb": "cuda:0",
        "base_model.model.model.visual": "cuda:0",
        "base_model.model.lm_head": "cuda:1",
        "base_model.model.model.language_model.norm": "cuda:1",
        **{f"base_model.model.model.language_model.layers.{i}": "cuda:0" for i in range(0, 18)},
        **{f"base_model.model.model.language_model.layers.{i}": "cuda:1" for i in range(18, 36)},
    }
    model.backbone = dispatch_model(model.backbone, device_map=device_map)
    optimizer = torch.optim.AdamW(model.parameters_(), lr=args.base_lr, betas=(args.beta_0, args.beta_1), eps=args.eps, weight_decay=args.wd)
    from custom_datasets import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.train_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
    # train

    resume(model, args)
    start_epoch = 0
    best_loss = float("inf")

    last_logging = time.time()
    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter("Loss", ":.4f")
        if is_distributed(): sampler.set_epoch(epoch)
        
        model.train()
        # lr = adjust_learning_rate(optimizer, epoch)
        lr = args.base_lr
        for step, (input_ids, labels_ids, payload_ids, position_ids, attention_mask, labels) in enumerate(loader, start=epoch*len(loader)):
            # forward
            optimizer.zero_grad()
            assert input_ids.shape[1] <= 4096
            result: Qwen3VLCausalLMOutputWithPast = model((input_ids, labels_ids, payload_ids, position_ids, attention_mask))
            # backward
            loss = result.loss
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), len(labels))
            if args.is_master and step % args.log_freq == 0:
                current_time = time.time()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {losses.avg:.4f}, Time: {current_time - last_logging:.2f}s")
                last_logging = current_time
                import sys
                sys.stdout.flush()
        # INSERT_YOUR_CODE
        # 输出每个GPU的显存用量
        if args.is_master:
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                print(f"GPU {i}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        # eval_loss, eval_acc, pre, rec, f1 = eval(args, model)
        eval_acc = 0.0 #测试
        if args.is_master:
            is_best = losses.avg < best_loss
            if is_best: best_loss = losses.avg
            if (epoch % args.save_freq == 0) or is_best:
                state = dict(epoch=epoch, model=model.state_dict_(args), optimizer=optimizer.state_dict(), best_loss=best_loss)
                save_checkpoint(state, is_best, args)
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=best_loss, lr=lr, eval_acc=eval_acc)
            print(json.dumps(logs))
            print(f"Epoch {epoch} => Eval accuracy ({eval_acc:.4f} %)")

    if is_distributed():
        dist.destroy_process_group()
    return

def eval(args, model = None):
    import sys

    if model is None:
        init_distributed(args)
        from model import ProposeModel
        model = ProposeModel(args)
        # print(model)
        # print("parameters:")
        # for name, param in model.named_parameters(recurse=True):
        #     print(name)
        # print("buffers:")
        # for name, buffer in model.named_buffers(recurse=True):
        #     print(name)
        # print(model.backbone.base_model.model.model.language_model.embed_tokens.weight.requires_grad)
        # return

        model.encoder = model.encoder.to('cuda:0')
        model.text_embedder = model.text_embedder.to('cuda:0')
        device_map = {
            "base_model.model.model.language_model.embed_tokens": "cuda:0",
            "base_model.model.model.language_model.rotary_emb": "cuda:0",
            "base_model.model.model.visual": "cuda:0",
            "base_model.model.lm_head": "cuda:1",
            "base_model.model.model.language_model.norm": "cuda:1",
            **{f"base_model.model.model.language_model.layers.{i}": "cuda:0" for i in range(0, 18)},
            **{f"base_model.model.model.language_model.layers.{i}": "cuda:1" for i in range(18, 36)},
        }
        model.backbone = dispatch_model(model.backbone, device_map=device_map)
        from custom_datasets import CustomDataset, collate_LLMDataset
        dataset = CustomDataset(args.test_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None

        resume(model, args)

    model.eval()
    # loss_list = []
    # for step, (input_ids, labels_ids, payload_ids, position_ids, attention_mask, labels) in enumerate(loader):
    #     # print(f"input_ids.shape: {input_ids.shape}")
    #     # sys.stdout.flush()
    #     assert input_ids.shape[1] <= 4096
    #     result: Qwen3VLCausalLMOutputWithPast = model((input_ids, labels_ids, payload_ids, position_ids, attention_mask))
    #     loss = result.loss
    #     loss_list.append(loss.item())
    # if len(loss_list) > 0:
    #     avg_loss = sum(loss_list) / len(loss_list)
    #     print(f"Eval Loss: avg={avg_loss:.4f} (samples={len(loss_list)})")
    # else:
    #     print("No valid samples for eval loss.")
    avg_loss = 0.0

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)

    # 初始化统计变量
    correct_count = 0
    total_count = 0
    tp = 0  # True Positive: 预测正确（is_same == True）
    fp = 0  # False Positive: 预测正确但实际错误（在这个场景中，如果is_same==True，说明预测正确，所以fp=0）
    fn = 0  # False Negative: 预测错误但实际正确（在这个场景中，如果is_same==False，说明预测错误，所以fn=is_same==False的数量）
    
    for step, (input_ids, labels_ids, payload_ids, position_ids, attention_mask, labels) in enumerate(loader):
        assert input_ids.shape[1] <= 4096
        # INSERT_YOUR_CODE
        first_not_minus_100 = (labels_ids[0] != -100).nonzero()[0].item()
        input_ids_ = input_ids[:, :first_not_minus_100]
        position_ids = position_ids[:, :first_not_minus_100]
        attention_mask = attention_mask[:, :first_not_minus_100]
        result = model((input_ids_, None, payload_ids, position_ids, attention_mask))
    # INSERT_YOUR_CODE
    # 比较result[0]和input_ids[0]的长度，若长度相等则比较是否每个位置都完全一样
        result = result.to('cpu')
        # from preprocess.utils import _ids_to_str
        # print("input_label_part:", _ids_to_str(input_ids[0][first_not_minus_100:], type="qwen3vl"))
        # print("output_label_part:", _ids_to_str(result[0], type="qwen3vl"))
        # print()
        # sys.stdout.flush()
        is_same = torch.equal(result[0], input_ids[0][first_not_minus_100:])
        # 统计
        total_count += 1
        if is_same:
            correct_count += 1
            tp += 1
        else:
            fn += 1
    
    # 计算指标
    if total_count > 0:
        acc = correct_count / total_count
    else:
        acc = 0.0
    
    # 精确率 (Precision): TP / (TP + FP)
    if tp + fp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0.0
    
    # 召回率 (Recall): TP / (TP + FN)
    if tp + fn > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0.0
    
    # F1分数: 2 * (Precision * Recall) / (Precision + Recall)
    if pre + rec > 0:
        f1 = 2 * (pre * rec) / (pre + rec)
    else:
        f1 = 0.0
    
    # 打印结果
    print(f"Evaluation Metrics:")
    print(f"  Total samples: {total_count}")
    print(f"  Correct samples: {correct_count}")
    print(f"  Accuracy (ACC): {acc:.4f}")
    print(f"  Precision (PRE): {pre:.4f}")
    print(f"  Recall (REC): {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return avg_loss, acc, pre, rec, f1

def add_args(parser):
    parser.add_argument('--finetune_mode', action='store_true', default=False, help="是否训练模式")
    parser.add_argument('--align1_mode', action='store_true', default=False, help="是否训练表格对齐")
    parser.add_argument('--align2_mode', action='store_true', default=False, help="是否训练载荷对齐")
    parser.add_argument('--eval_mode', action='store_true', default=False, help="是否评估")
    parser.add_argument('--train_dir', type=str, default='/datasets/train/', help="训练数据目录")
    parser.add_argument('--test_dir', type=str, default='/datasets/test/', help="测试数据目录")
    parser.add_argument('--save_dir', type=str, default='./out/', help="模型保存目录")
    parser.add_argument('--log_freq', type=int, default=5, help="日志打印频率")
    parser.add_argument('--save_freq', type=int, default=10, help="模型保存频率（每多少个epoch保存一次）")
    parser.add_argument('--resume_encoder', type=str, default="", help="是否从checkpoint恢复encoder")
    parser.add_argument('--resume_linear', type=str, default="", help="是否从checkpoint恢复linear层")
    parser.add_argument('--resume_lora0', type=str, default="", help="是否从checkpoint恢复lora0")
    parser.add_argument('--resume_lora1', type=str, default="", help="是否从checkpoint恢复lora1")
    parser.add_argument('--resume_log', action='store_true', default=False, help="是否输出resume日志")

    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=256, help="总batch size（会被world_size整除）")
    parser.add_argument('--seed', type=int, default=None, help="随机种子")
    parser.add_argument('--base_lr', type=float, default=0.05, help="基础学习率")
    parser.add_argument('--beta_0', type=float, default=0.9, help="AdamW中的beta_0参数")
    parser.add_argument('--beta_1', type=float, default=0.999, help="AdamW中的beta_1参数")
    parser.add_argument('--eps', type=float, default=1e-8, help="AdamW中的eps参数")
    parser.add_argument('--wd', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--fix_pred_lr', action='store_true', default=True, help="预测头是否固定学习率")

    parser.add_argument('--projector', type=str, default='linear', help="投影头类型")
    parser.add_argument('--projector_arch', type=str, default='2048-2048', help="投影头结构")
    parser.add_argument('--linear_output_dim', type=int, default=4096, help="线性投影头的输出维度")
    parser.add_argument('--llm', type=str, default='Qwen3VL-3B-Instruct', help="LLM模型名字")
    
    parser.add_argument('--workers', type=int, default=1, help="dataloader线程数")
    parser.add_argument('--device', type=str, default='cuda', help="设备类型(cpu, cuda, mps)")
    parser.add_argument('--nodistributed', dest='distributed', action='store_false', default=True, help="禁用分布式训练")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    if args.eval_mode:
        eval(args, None)
    elif args.finetune_mode or args.align1_mode or args.align2_mode:
        train(args)
    else:
        raise ValueError("Invalid mode!")