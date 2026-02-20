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
        args.per_device_batch_size_t = getattr(args, 'batch_size_t', args.batch_size)
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
        args.per_device_batch_size_t = int(getattr(args, 'batch_size_t', args.batch_size) / torch.cuda.device_count())
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

class MemoryBank:
    """Memory Bank 用于存储最近的源域和目标域样本特征"""
    def __init__(self, queue_size: int, device: str = 'cuda:1'):
        self.queue_size = queue_size
        self.device = device
        # 源域队列: list of (hidden_states, labels)
        self.src_queue = []
        # 目标域队列: list of (hidden_states, logits)
        self.tgt_queue = []
    
    def update(self, src_samples: list, tgt_samples: list):
        """更新队列，添加新样本（detach后），移除最旧的样本"""
        # 添加新样本（detach以断开计算图）
        for hidden, labels in src_samples:
            self.src_queue.append((hidden.detach().to(self.device), labels.detach().to(self.device) if isinstance(labels, torch.Tensor) else labels))
        for hidden, logits in tgt_samples:
            self.tgt_queue.append((hidden.detach().to(self.device), logits.detach().to(self.device)))
        
        # 移除超出队列大小的旧样本
        while len(self.src_queue) > self.queue_size:
            self.src_queue.pop(0)
        while len(self.tgt_queue) > self.queue_size:
            self.tgt_queue.pop(0)
    
    def get_samples(self, current_src: list = None, current_tgt: list = None):
        """获取用于MMD计算的样本，包括队列中的历史样本和当前样本"""
        src_samples = list(self.src_queue)
        tgt_samples = list(self.tgt_queue)
        if current_src:
            src_samples.extend(current_src)
        if current_tgt:
            tgt_samples.extend(current_tgt)
        return src_samples, tgt_samples
    
    def __len__(self):
        return len(self.src_queue)

def get_grad(model):
    g = []
    none_count = 0
    none_positions = []
    for i, p in enumerate(model.parameters_()):
        if p.grad is None:
            none_count += 1
            none_positions.append(i)
            g.append(torch.zeros_like(p).view(-1).contiguous().to('cuda:1'))
        else:
            g.append(torch.as_tensor(p.grad).view(-1).contiguous().to('cuda:1'))
    
    if none_count != 0 and none_count != 2:
        raise RuntimeError(f"Too many parameters with None gradient: {none_count}, positions: {none_positions}")
    if none_count != 0 and set(none_positions) != {288, 289}:
        raise RuntimeError(f"Parameters with None gradient at unexpected positions: {none_positions}, expected [288, 289]")
    
    return torch.cat(g, dim=0).detach()

def to_grad_and_set(model, g):
    """将 get_grad 输出的扁平化梯度向量还原为每个参数对应的梯度列表"""
    grads = []
    offset = 0
    for p in model.parameters_():
        numel = p.numel()
        grad = g[offset:offset + numel].view(p.shape).to(p.device)
        p.grad = grad
        offset += numel
    return

def compute_and_fuse_grads(model, optimizer, dom_loss_fn, gh_fn, src_samples, tgt_samples, dom_loss_weight):
    """计算 MMD loss 并与 CE loss 梯度融合，返回融合后的梯度"""
    ce_grads = get_grad(model)
    optimizer.zero_grad()
    dom_loss_ = dom_loss_fn(src_samples, tgt_samples) * dom_loss_weight
    dom_loss_.backward()
    del dom_loss_
    dom_loss_grads = get_grad(model)
    optimizer.zero_grad()
    gh_grads = gh_fn(ce_grads, dom_loss_grads)
    return gh_grads

# TODO: 调整优化器参数 & 添加scheduler & 并行训练
def train(args):
    # init
    init_distributed(args)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.is_master:
        for k, v in vars(args).items():
            print(f"{k}: {v}")
            
    args.labels = _get_labels(args.train_dir)
    # load model
    from z1.model import ProposeModel
    model = ProposeModel(args)
    # print(model)
    for name in model.named_parameters_names(True):
        print(name)
    # for name in model.parameters_():
    #     print(name)
    # for name in model.parameters():
    #     print(name)
    # return
    
    from dataset import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.train_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
    
    if args.z2_mode:
        dataset_t = CustomDataset(args.train_dir_t)
        loader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.per_device_batch_size_t, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
    model.dispatch(split_layers_num=args.split_layers_num, single_gpu=args.single_gpu)

    args.optimizer = torch.optim.AdamW(model.parameters_(), lr=args.base_lr, betas=(args.beta_0, args.beta_1), eps=args.eps, weight_decay=args.wd)
    # 混合精度 GradScaler
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler(enabled=(args.amp and amp_dtype == torch.float16))
    args.start_epoch = 0
    args.best_loss = float("inf")
    args.best_acc = float("-inf")
    model.resume(args)
    if args.amp:
        print(f"Mixed precision (AMP) enabled, dtype={args.amp_dtype}.")
    # 清理缓存以释放加载优化器状态后可能产生的内存碎片
    torch.cuda.empty_cache()
    
    if args.reduce_on_plateau:
        from transformers import get_reduce_on_plateau_schedule
        scheduler = get_reduce_on_plateau_schedule(args.optimizer, mode='min', factor=0.1, patience=0)
    if args.constant:
        from transformers import get_constant_schedule
        scheduler = get_constant_schedule(args.optimizer)
    else:
        total_steps = args.epochs * len(loader) // args.accumulation_steps
        warmup_steps = args.epochs * len(loader) * args.warmup // args.accumulation_steps
        passed_steps = args.start_epoch * len(loader) // args.accumulation_steps
        if args.error_scheduler:
            passed_steps = (args.epochs-args.start_epoch) * len(loader) // args.accumulation_steps
        if args.ignore_passed:
            passed_steps = 0
        warmup_steps = max(0, warmup_steps - passed_steps)
        total_steps -= passed_steps
        # 此处并不能够按cosine从半路开始，而是开始total_steps为整个的cosine
        if args.min_lr is None:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                args.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=args.num_cycles
            )
        else:
            from transformers import get_cosine_with_min_lr_schedule_with_warmup
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                args.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=args.num_cycles,
                min_lr=args.min_lr
            )

        print(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, passed_steps={passed_steps}")
        import sys; sys.stdout.flush()
    # z2
    if args.z2_mode:
        from utils.gh import get_GH
        gh = get_GH(args.gh)
        if args.gh == "gh++":
            gh = lambda g1, g2: gh(g1, g2, lam=args.lambda_)
        from utils.mmd import get_MMD
        dom_loss = get_MMD(args.dom_loss, args.weight_type)
        # 初始化 memory bank（如果启用）
        memory_bank = MemoryBank(args.queue_size) if args.memory_bank else None
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
        if args.z2_mode:
            accum_src_samples = []
            accum_tgt_samples = []
            accum_grads = None
        train_iter = enumerate(zip(loader, loader_t), start=epoch*len(loader)) if args.z2_mode else enumerate(loader, start=epoch*len(loader))
        for step, batch_data in train_iter:
            if args.z2_mode:
                src_batch, tgt_batch = batch_data
                input_ids, labels_ids, payloads, position_ids, attention_mask, labels, _ = src_batch
                input_ids_t, labels_ids_t, payloads_t, position_ids_t, attention_mask_t, labels_t, _ = tgt_batch
                # print(f"Processing batch {step}: src_labels={labels}, tgt_labels={labels_t}")
            else:
                input_ids, labels_ids, payloads, position_ids, attention_mask, labels, _ = batch_data
            try:
                # forward
                assert input_ids.shape[1] <= 4096
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                    result: Qwen3VLCausalLMOutputWithPast = model(
                        input_ids=input_ids,
                        labels=labels_ids if not args.z2_mode else None,
                        payloads=payloads,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        classifier_labels=labels if args.z2_mode else None,
                        rope_deltas=None
                    )
                    if args.z2_mode:
                        result2: Qwen3VLCausalLMOutputWithPast = model(
                            input_ids=input_ids_t,
                            labels=None,
                            payloads=payloads_t,
                            position_ids=position_ids_t,
                            attention_mask=attention_mask_t,
                            classifier_labels=None,
                            rope_deltas=None
                        )
                # backward
                loss = result.loss / args.accumulation_steps
                losses.update(loss.item(), len(labels))
                
                # z2 模式处理
                if args.z2_mode:
                    # 当前 batch 的样本
                    src_labels = torch.tensor([model.label2id[x] for x in labels], device=result.last_hidden_states.device)
                    current_src = (result.last_hidden_states, src_labels)
                    current_tgt = (result2.last_hidden_states, result2.logits)
                    
                    # 累积样本用于 accumulation 模式
                    if memory_bank is None:
                        accum_src_samples.append(current_src)
                        accum_tgt_samples.append(current_tgt)
                        scaler.scale(loss).backward(retain_graph=True)
                    # memory_bank 模式：每个 batch 计算 MMD
                    else:
                        mb_src, mb_tgt = memory_bank.get_samples([current_src], [current_tgt])
                        scaler.scale(loss).backward(retain_graph=True)
                        print(f"Memory bank size: src={len(mb_src)}, tgt={len(mb_tgt)}")
                        
                        if len(mb_src) > 1:
                            grads = compute_and_fuse_grads(model, args.optimizer, dom_loss, gh, mb_src, mb_tgt, args.dom_loss_weight)
                            accum_grads = grads if accum_grads is None else accum_grads + grads
                        # 更新 memory bank
                        if args.momentum_k > 0:
                            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                                k_result = model.forward_k(input_ids, attention_mask, position_ids, payloads)
                                k_result_t = model.forward_k(input_ids_t, attention_mask_t, position_ids_t, payloads_t)
                            memory_bank.update([(k_result.last_hidden_states, src_labels)], [(k_result_t.last_hidden_states, k_result_t.logits)])
                        else:
                            memory_bank.update([(current_src[0].detach(), current_src[1].detach())], [(current_tgt[0].detach(), current_tgt[1].detach())])
                    del result, result2
                else:
                    # 非 z2 模式
                    del result
                    scaler.scale(loss).backward()
                del loss
                
                # accumulation step 处理
                is_accum_step = (step + 1) % args.accumulation_steps == 0
                if is_accum_step:
                    if args.z2_mode:
                        # 无 memory_bank：在 accumulation step 计算 MMD
                        if memory_bank is None:
                            # 获取用于 MMD 的样本
                            mmd_src, mmd_tgt = accum_src_samples, accum_tgt_samples
                            accum_grads = compute_and_fuse_grads(model, args.optimizer, dom_loss, gh, mmd_src, mmd_tgt, args.dom_loss_weight)
                            # 清空累积样本
                            del accum_src_samples, accum_tgt_samples
                            accum_src_samples = []
                            accum_tgt_samples = []

                        to_grad_and_set(model, accum_grads)

                    scaler.step(args.optimizer)
                    scaler.update()
                    # if args.z2_mode and args.momentum_k > 0:
                    #     model.momentum_update_k_encoder()
                    if args.reduce_on_plateau:
                        decrease_losses = losses.avg - last_losses if last_losses is not None else 0
                        scheduler.step(decrease_losses)
                    else:
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
                lr = scheduler.get_last_lr()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {losses.avg:.4f}, Time: {current_time - last_logging:.2f}s", "LR: ", lr)
                last_logging = current_time
                import sys
                sys.stdout.flush()
        
        # 在eval前清理缓存，释放训练过程中的内存
        torch.cuda.empty_cache()
        full_eval = (args.epochs - epoch) <= args.full_eval_epochs
        eval_loss, eval_acc, pre, rec, f1 = eval(args, model, loss_only=not full_eval)
        # eval后清理缓存，释放eval过程中的内存
        torch.cuda.empty_cache()
        # eval_acc = 0.0 #测试
        if args.is_master:
            is_best = (eval_acc > args.best_acc or (eval_acc == args.best_acc and eval_loss < args.best_loss)) if full_eval else (eval_loss < args.best_loss)
            if is_best: args.best_loss = losses.avg; args.best_acc = eval_acc
            state = dict(epoch=epoch, model=model.state_dict_(args), optimizer=args.optimizer.state_dict(), 
                best_loss=args.best_loss, best_acc=args.best_acc, scaler=scaler.state_dict())
            save_checkpoint(state, is_best, args, filename=f"checkpoint-{epoch}.pt")
            lr = [param_group['lr'] for param_group in args.optimizer.param_groups]
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=args.best_loss, best_acc=args.best_acc, eval_acc=eval_acc, lr=lr)
            print(json.dumps(logs))
            print(f"Epoch {epoch} => Eval accuracy ({eval_acc:.4f} %)")

    if is_distributed():
        dist.destroy_process_group()
    return

def _compute_cls_metrics(all_preds, all_gts, class_name_fn=str):
    """计算多分类指标：acc, macro precision, macro recall, macro f1，并打印详细信息"""
    total_count = len(all_gts)
    correct_count = sum(p == g for p, g in zip(all_preds, all_gts))
    acc = correct_count / total_count if total_count > 0 else 0.0

    all_classes = sorted(set(all_gts + all_preds))
    per_class = {}
    for c in all_classes:
        tp = sum(1 for p, g in zip(all_preds, all_gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(all_preds, all_gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(all_preds, all_gts) if p != c and g == c)
        pre_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * pre_c * rec_c / (pre_c + rec_c) if (pre_c + rec_c) > 0 else 0.0
        per_class[c] = (pre_c, rec_c, f1_c)

    n = len(per_class)
    pre = sum(v[0] for v in per_class.values()) / n if n else 0.0
    rec = sum(v[1] for v in per_class.values()) / n if n else 0.0
    f1  = sum(v[2] for v in per_class.values()) / n if n else 0.0

    print(f"  Total samples: {total_count}")
    print(f"  Correct samples: {correct_count}")
    print(f"  Accuracy (ACC): {acc:.4f}")
    print(f"  Macro Precision (PRE): {pre:.4f}")
    print(f"  Macro Recall (REC): {rec:.4f}")
    print(f"  Macro F1 Score: {f1:.4f}")
    for c, (p, r, f) in per_class.items():
        print(f"    Class '{class_name_fn(c)}': PRE={p:.4f}, REC={r:.4f}, F1={f:.4f}")

    return acc, pre, rec, f1

def _get_labels(dataset_dir, mode="train"):
    import re
    from pathlib import Path
    known_labels = set()
    for pkl_file in Path(dataset_dir).glob("*.pkl"):
        m = re.search(rf'{mode}_(.+?)_part', pkl_file.name)
        if m:
            known_labels.add(m.group(1))
    return known_labels

def eval(args, model = None, loss_only = False, calculate_loss = True):
    import sys

    if model is None:
        init_distributed(args)
        from z1.model import ProposeModel
        model = ProposeModel(args)
        model.dispatch(split_layers_num=args.split_layers_num)
        model.resume(args)
    else:
        model.backbone.gradient_checkpointing_disable()
        model.backbone.disable_input_require_grads()
    model.eval()

    from dataset import CustomDataset, collate_LLMDataset
    dataset = CustomDataset(args.test_dir)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    if calculate_loss:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None


        # ===== 第一次遍历：计算loss（+ z2_mode下收集分类预测） =====
        loss_list = []
        all_preds = []
        all_gts = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels, rope_deltas) in enumerate(loader):
                assert input_ids.shape[1] <= 4096
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                    result: Qwen3VLCausalLMOutputWithPast = model(
                        input_ids=input_ids,
                        labels=labels_ids if not args.z2_mode else None,
                        payloads=payloads,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        classifier_labels=labels if args.z2_mode else None,
                        rope_deltas=None
                    )
                if args.z2_mode:
                    loss_list.append(result["loss"].item())
                    all_preds.extend(result["logits"].argmax(dim=-1).cpu().tolist())
                    all_gts.extend(model.label2id[x] for x in labels)
                else:
                    loss_list.append(result.loss.item())

        avg_loss = sum(loss_list) / len(loss_list) if loss_list else float("inf")
        print(f"Eval Loss: avg={avg_loss:.4f} (samples={len(loss_list)})" if loss_list else "No valid samples for eval loss.")

        if args.z2_mode:
            # z2_mode: 直接根据classifier logits计算分类指标
            id2label = {v: k for k, v in model.label2id.items()}
            print(f"Evaluation Metrics (z2 classification):")
            acc, pre, rec, f1 = _compute_cls_metrics(all_preds, all_gts, class_name_fn=lambda c: id2label.get(c, c))
            torch.cuda.empty_cache()
            return avg_loss, acc, pre, rec, f1

        if loss_only:
            torch.cuda.empty_cache()
            return avg_loss, 0.0, 0.0, 0.0, 0.0

    # ===== 第二次遍历：生成并匹配label =====
    # 从test_dir的pkl文件名中提取label集合（train_XXX_part模式）
    known_labels = _get_labels(args.test_dir, 'test')
    UNKNOWN_LABEL = "__unknown__"
    print(f"Extracted labels from pkl filenames: {known_labels}")

    from preprocess.utils import _str_to_ids, _ids_to_str
    max_label_len = max(len(_str_to_ids(lbl, type="qwen3vl")[0]) for lbl in known_labels) + 1 if known_labels else 0
    print(f"Max label token length: {max_label_len}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_LLMDataset)

    try:
        total_steps = len(loader)
    except TypeError:
        total_steps = None

    all_preds = []
    all_gts = []
    unknown_count = 0

    with torch.no_grad():
        for step, (input_ids, labels_ids, payloads, position_ids, attention_mask, labels, rope_deltas) in enumerate(loader):
            if total_steps is None:
                progress = f"[{step + 1}]"
            else:
                progress = f"[{step + 1}/{total_steps}]"
            assert input_ids.shape[1] <= 4096
            first_not_minus_100 = (labels_ids[0] != -100).nonzero()[0].item()
            input_ids_ = input_ids[:, :first_not_minus_100]
            position_ids = position_ids[:, :, :first_not_minus_100]
            attention_mask = attention_mask[:, :first_not_minus_100]
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                result = model.generate(
                    input_ids=input_ids_,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    payloads=payloads,
                    rope_deltas=rope_deltas if not args.not_use_rope_deltas else None,
                    max_new_tokens=max_label_len + 5,
                    use_cache=True,
                    do_sample=False
                )
            result = result.to('cpu')
            gt_text = _ids_to_str(input_ids[0][first_not_minus_100:], type="qwen3vl").strip().removesuffix('<|im_end|>')
            pred_text = _ids_to_str(result[0][first_not_minus_100:], type="qwen3vl").strip().removesuffix('<|im_end|>')
            pred_label = pred_text if pred_text in known_labels else UNKNOWN_LABEL
            print(progress, "label:", gt_text)
            gt_label = gt_text

            print(progress, f"GT: '{gt_label}', Pred: '{pred_text}' -> '{pred_label}', Match: {gt_label == pred_label}")
            print(progress, "==========================================================")
            sys.stdout.flush()

            all_gts.append(gt_label)
            all_preds.append(pred_label)
            if pred_label == UNKNOWN_LABEL:
                unknown_count += 1

    print(f"Evaluation Metrics:")
    print(f"  Unknown predictions: {unknown_count}")
    acc, pre, rec, f1 = _compute_cls_metrics(all_preds, all_gts)

    # eval结束后清理缓存
    torch.cuda.empty_cache()
    return avg_loss, acc, pre, rec, f1

def add_args(parser):
    #mode
    parser.add_argument('--finetune_mode', action='store_true', default=False, help="是否训练模式")
    parser.add_argument('--align1_mode', action='store_true', default=False, help="是否训练表格对齐")
    parser.add_argument('--align2_mode', action='store_true', default=False, help="是否训练载荷对齐")
    parser.add_argument('--eval_mode', action='store_true', default=False, help="是否评估")
    parser.add_argument('--test_mode', action='store_true', default=False, help="是否测试")
    parser.add_argument('--z2_mode', action='store_true', default=False, help="是否z2模式")
    parser.add_argument('--wo_weight_mode', action='store_true', default=False)
    #z2
    parser.add_argument('--gh', type=str, default='deactivated', choices=['gh', 'gh++', 'deactivated'], help="GH模式选择")
    parser.add_argument('--lambda_', type=float, default=0.5, help="GH++模式lambda")
    parser.add_argument('--dom_loss', type=str, default='mmd', choices=['mmd', 'lmmd'], help="域损失类型")
    parser.add_argument('--dom_loss_weight', type=float, default=1.0, help="域分类损失权重")
    parser.add_argument('--weight_type', type=str, default='softmax', choices=['a-softmax', 'average'], help="域损失类型")
    # memory bank 相关参数
    parser.add_argument('--memory_bank', action='store_true', default=False, help="是否启用memory bank模式")
    parser.add_argument('--momentum_k', type=float, default=0.0, help="momentum k-encoder动量系数，0表示不使用k-encoder")
    parser.add_argument('--queue_size', type=int, default=256, help="memory bank队列大小")
    #resume & dir
    parser.add_argument('--train_dir', type=str, default='/datasets/train/', help="训练数据目录")
    parser.add_argument('--train_dir_t', type=str, default='/datasets/train_t/', help="z2模式目标域训练数据目录")
    parser.add_argument('--batch_size_t', type=int, default=1, help="z2模式目标域batch size")
    parser.add_argument('--test_dir', type=str, default='/datasets/test/', help="测试数据目录")
    parser.add_argument('--save_dir', type=str, default='./out/', help="模型保存目录")
    parser.add_argument('--log_freq', type=int, default=16, help="日志打印频率")
    parser.add_argument('--save_freq', type=int, default=10, help="模型保存频率（每多少个epoch保存一次）")
    parser.add_argument('--full_eval_epochs', type=int, default=3, help="最后多少个epoch进行完整eval（含generate推理），其余epoch仅计算eval loss")
    parser.add_argument('--resume_encoder', type=str, default="", help="是否从checkpoint恢复encoder")
    parser.add_argument('--resume_linear', type=str, default="", help="是否从checkpoint恢复linear层")
    parser.add_argument('--resume_lora0', type=str, default="", help="是否从checkpoint恢复lora0")
    parser.add_argument('--resume_lora1', type=str, default="", help="是否从checkpoint恢复lora1")
    parser.add_argument('--resume_log', action='store_true', default=False, help="是否输出resume日志")
    parser.add_argument('--not_use_rope_deltas', action='store_true', default=False, help="是否不使用rope deltas")
    #train
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
    parser.add_argument('--error_scheduler', action='store_true', default=False, help="是否错误scheduler")
    parser.add_argument('--ignore_passed', action='store_true', default=False, help="是否忽略已通过的样本")
    parser.add_argument('--constant', action='store_true', default=False, help="是否使用constant scheduler")
    parser.add_argument('--reduce_on_plateau', action='store_true', default=False, help="是否reduce on plateau")
    parser.add_argument('--num_cycles', type=float, default=0.5, help="Warm up value.")
    parser.add_argument('--min_lr', type=float, default=None, help="最小学习率")

    # INSERT_YOUR_CODE
    parser.add_argument('--accumulation_steps', type=int, default=8, help="梯度累积步数")

    parser.add_argument('--projector', type=str, default='linear', help="投影头类型")
    # parser.add_argument('--projector_arch', type=str, default='768-4096', help="投影头结构")
    parser.add_argument('--linear_output_dim', type=int, default=4096, help="线性投影头的输出维度")
    parser.add_argument('--llm', type=str, default='Qwen3VL-3B-Instruct', help="LLM模型名字")
    
    parser.add_argument('--workers', type=int, default=1, help="dataloader线程数")
    parser.add_argument('--device', type=str, default='cuda', help="设备类型(cpu, cuda, mps)")
    parser.add_argument('--nodistributed', dest='distributed', action='store_false', default=True, help="禁用分布式训练")
    parser.add_argument('--amp', action='store_true', default=False, help="启用混合精度训练(AMP)")
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help="混合精度数据类型(bf16或fp16)")
    parser.add_argument('--tf32', action='store_true', default=False, help="启用TF32加速(仅Ampere及以上GPU)")
    parser.add_argument('--split_layers_num', type=int, default=25, help="split layers num")
    parser.add_argument('--compile', action='store_true', default=False, help="启用torch.compile加速")
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'], help="torch.compile模式")
    parser.add_argument('--single_gpu', action='store_true', default=False, help="单显卡模式，所有模块放到cuda:0")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()

    if args.eval_mode or args.test_mode:
        eval(args, None, False, False)
    elif args.finetune_mode or args.align1_mode or args.align2_mode:
        train(args)
    else:
        raise ValueError("Invalid mode!")