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

# TODO: 调整优化器参数 & 添加scheduler & 并行训练
def train(args):
    # init
    init_distributed(args)
    if args.is_master:
        for k, v in vars(args).items():
            print(f"{k}: {v}")
            
    from z1.model import ProposeModel
    model = ProposeModel(args)
    print(model)
    # return

    model.device = torch.device('cuda:0')
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
    from dataset import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.train_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
    model.backbone = dispatch_model(model.backbone, device_map=device_map)

    args.optimizer = torch.optim.AdamW(model.parameters_(), lr=args.base_lr, betas=(args.beta_0, args.beta_1), eps=args.eps, weight_decay=args.wd)
    # 混合精度 GradScaler
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler(enabled=(args.amp and amp_dtype == torch.float16))
    args.start_epoch = 0
    args.best_loss = float("inf")
    args.best_acc = float("-inf")
    model.resume(args)
    # 恢复 scaler 状态（如果 checkpoint 中存在）
    if hasattr(args, '_scaler_state_dict') and args._scaler_state_dict is not None:
        scaler.load_state_dict(args._scaler_state_dict)
    print("start_epoch:", args.start_epoch, "best_loss:", args.best_loss, "best_acc:", args.best_acc, "optimizer:", args.optimizer.state_dict()["param_groups"])
    if args.amp:
        print(f"Mixed precision (AMP) enabled, dtype={args.amp_dtype}.")
    # 清理缓存以释放加载优化器状态后可能产生的内存碎片
    torch.cuda.empty_cache()
    
    from transformers import get_cosine_schedule_with_warmup
    total_steps = args.epochs * len(loader) // args.accumulation_steps
    warmup_steps = args.epochs * len(loader) * args.warmup // args.accumulation_steps
    passed_steps = (args.epochs-args.start_epoch) * len(loader) // args.accumulation_steps
    warmup_steps = max(0, warmup_steps - passed_steps)
    total_steps -= passed_steps
    scheduler = get_cosine_schedule_with_warmup(
        args.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # train

    last_logging = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter("Loss", ":.4f")
        if is_distributed(): sampler.set_epoch(epoch)

        # 每个epoch开始时清理缓存，减少内存碎片
        torch.cuda.empty_cache()
        
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
        model.train()
        args.optimizer.zero_grad()
        for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels) in enumerate(loader, start=epoch*len(loader)):
            try:
                # forward
                assert input_ids.shape[1] <= 4096
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                    result: Qwen3VLCausalLMOutputWithPast = model(
                        input_ids=input_ids,
                        labels=labels_ids,
                        payloads=payloads,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        rope_deltas=None
                    )
                # backward
                loss = result.loss / args.accumulation_steps
                # 显式删除result以释放内存
                del result
                scaler.scale(loss).backward()
                losses.update(loss.item(), len(labels))
                del loss
                if (step + 1) % args.accumulation_steps == 0:
                    scaler.step(args.optimizer)
                    scaler.update()
                    scheduler.step()
                    args.optimizer.zero_grad()
                    # 在优化器步骤后清理缓存，释放梯度累积占用的内存
                    torch.cuda.empty_cache()
                # 定期清理缓存，防止内存碎片化累积（每100步清理一次）
                elif (step + 1) % 50 == 0:
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                # OOM异常捕获和自动恢复
                if args.is_master:
                    import sys
                    print(f"\n{'='*60}", file=sys.stderr)
                    print(f"OOM Error detected at Epoch {epoch}, Step {step}", file=sys.stderr)
                    print(f"Sequence length: {input_ids.shape[1]}", file=sys.stderr)
                    print(f"Batch size: {input_ids.shape[0]}", file=sys.stderr)
                    print(f"Skipping this batch and continuing training...", file=sys.stderr)
                    print(f"{'='*60}\n", file=sys.stderr)
                    sys.stderr.flush()
                
                # 清理所有可能的缓存
                torch.cuda.empty_cache()
                # 确保梯度被清零，避免影响后续训练
                args.optimizer.zero_grad()
                # 跳过当前batch，继续训练
                continue
                
            if args.is_master and step % args.log_freq == 0:
                current_time = time.time()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {losses.avg:.4f}, Time: {current_time - last_logging:.2f}s")
                last_logging = current_time
                import sys
                sys.stdout.flush()
        
        # 在eval前清理缓存，释放训练过程中的内存
        torch.cuda.empty_cache()
        eval_loss, eval_acc, pre, rec, f1 = eval(args, model)
        # eval后清理缓存，释放eval过程中的内存
        torch.cuda.empty_cache()
        # eval_acc = 0.0 #测试
        if args.is_master:
            is_best =  eval_acc > args.best_acc or (eval_acc == args.best_acc and losses.avg < args.best_loss)
            if is_best: args.best_loss = losses.avg; args.best_acc = eval_acc
            if (epoch % args.save_freq == 0) or is_best:
                state = dict(epoch=epoch, model=model.state_dict_(args), optimizer=args.optimizer.state_dict(), 
                    best_loss=args.best_loss, best_acc=args.best_acc, scaler=scaler.state_dict())
                save_checkpoint(state, is_best, args)
            lr = [param_group['lr'] for param_group in args.optimizer.param_groups]
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=args.best_loss, best_acc=args.best_acc, eval_acc=eval_acc, lr=lr)
            print(json.dumps(logs))
            print(f"Epoch {epoch} => Eval accuracy ({eval_acc:.4f} %)")

    if is_distributed():
        dist.destroy_process_group()
    return

def eval(args, model = None):
    import sys

    if model is None:
        init_distributed(args)
        from z1.model import ProposeModel
        model = ProposeModel(args)
        print(model)
        model.device = torch.device('cuda:0')
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
        if args.test_mode:
            new_device_map = {}
            for k, v in device_map.items():
                new_device_map[k[len("base_model.model."):]] = v
            device_map = new_device_map
        model.backbone = dispatch_model(model.backbone, device_map=device_map)
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()

        model.resume(args)
    model.backbone.gradient_checkpointing_disable()
    model.backbone.disable_input_require_grads()
    model.eval()

    from dataset import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.test_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None

    loss_list = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels) in enumerate(loader):
            assert input_ids.shape[1] <= 4096
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
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

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)

    # 初始化统计变量
    correct_count = 0
    total_count = 0
    tp = 0  # True Positive: 预测正确（is_same == True）
    fp = 0  # False Positive: 预测正确但实际错误（在这个场景中，如果is_same==True，说明预测正确，所以fp=0）
    fn = 0  # False Negative: 预测错误但实际正确（在这个场景中，如果is_same==False，说明预测错误，所以fn=is_same==False的数量）
    
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
            max_new_tokens=512,
            do_sample=False
        )
        # INSERT_YOUR_CODE
        # 比较result[0]和input_ids[0]的长度，若长度相等则比较是否每个位置都完全一样
        result = result.to('cpu')
        from preprocess.utils import _ids_to_str
        # print("input_ids_length:", input_ids.shape[1])
        print("input_label_part:", _ids_to_str(input_ids[0][first_not_minus_100:], type="qwen3vl"))
        # print("input_total_part:", _ids_to_str(input_ids[0][first_not_minus_100:input_ids.shape[1]], type="qwen3vl"))
        # print("result_length:", result.shape[1])
        print("ouput_label_part:", _ids_to_str(result[0][first_not_minus_100:input_ids.shape[1]], type="qwen3vl"))
        # print("ouput_total_part:", _ids_to_str(result[0][first_not_minus_100:], type="qwen3vl"))
        is_same = torch.equal(result[0][first_not_minus_100:input_ids.shape[1]], input_ids[0][first_not_minus_100:])
        print("is_same:", "true" if is_same else "false")
        print("==========================================================")
        sys.stdout.flush()
        # 统计
        total_count += 1
        if is_same:
            correct_count += 1
            tp += 1
        else:
            fn += 1
        # 显式删除变量以释放内存
        del result, input_ids_, input_ids, labels_ids, payloads, position_ids, attention_mask, labels, rope_deltas
        # 每10个样本清理一次缓存，防止内存累积
        if (step + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
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
    
    # eval结束后清理缓存
    torch.cuda.empty_cache()
    return avg_loss, acc, pre, rec, f1

def add_args(parser):
    parser.add_argument('--finetune_mode', action='store_true', default=False, help="是否训练模式")
    parser.add_argument('--align1_mode', action='store_true', default=False, help="是否训练表格对齐")
    parser.add_argument('--align2_mode', action='store_true', default=False, help="是否训练载荷对齐")
    parser.add_argument('--eval_mode', action='store_true', default=False, help="是否评估")
    parser.add_argument('--test_mode', action='store_true', default=False, help="是否测试")
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

    parser.add_argument('--seed', type=int, default=None, help="随机种子")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=2, help="总batch size（会被world_size整除）")
    parser.add_argument('--base_lr', type=float, default=2e-5, help="基础学习率")
    parser.add_argument('--beta_0', type=float, default=0.9, help="AdamW中的beta_0参数")
    parser.add_argument('--beta_1', type=float, default=0.999, help="AdamW中的beta_1参数")
    parser.add_argument('--eps', type=float, default=1e-8, help="AdamW中的eps参数")
    parser.add_argument('--wd', type=float, default=1e-4, help="权重衰减")

    #scheduler
    parser.add_argument('--warmup', type=float, default=0.1, help="Warm up value.")
    # INSERT_YOUR_CODE
    parser.add_argument('--accumulation_steps', type=int, default=16, help="梯度累积步数")

    parser.add_argument('--projector', type=str, default='linear', help="投影头类型")
    # parser.add_argument('--projector_arch', type=str, default='768-4096', help="投影头结构")
    parser.add_argument('--linear_output_dim', type=int, default=4096, help="线性投影头的输出维度")
    parser.add_argument('--llm', type=str, default='Qwen3VL-3B-Instruct', help="LLM模型名字")
    
    parser.add_argument('--workers', type=int, default=1, help="dataloader线程数")
    parser.add_argument('--device', type=str, default='cuda', help="设备类型(cpu, cuda, mps)")
    parser.add_argument('--nodistributed', dest='distributed', action='store_false', default=True, help="禁用分布式训练")
    parser.add_argument('--amp', action='store_true', default=False, help="启用混合精度训练(AMP)")
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help="混合精度数据类型(bf16或fp16)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()

    if args.eval_mode or args.test_mode:
        eval(args, None)
    elif args.finetune_mode or args.align1_mode or args.align2_mode:
        train(args)
    else:
        raise ValueError("Invalid mode!")