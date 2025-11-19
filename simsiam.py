import json
import os
import math
import shutil
import random
import time
import sys

import torch
import torch.nn.functional as F

from dataclasses import dataclass
# from PIL import ImageFilter

from torch import nn
from torch import distributed as dist
# from torchvision import models, transforms, datasets

#############################################
#                Distributed                #
#############################################

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

#############################################
#                Dataloader                 #
#############################################

# class gaussian_blur:

#     def __init__ (self, sigma):
#         self.sigma = sigma
#     def __call__(self, x):
#         return x.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))

# class augment:
#     """MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709"""
#     def __init__(self):
#         self.transform = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomApply([gaussian_blur([.1, 2.])], p=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __call__(self, x):
#         x1, x2 = self.transform(x), self.transform(x)
#         return x1, x2

#############################################
#              Model Components             #
#############################################

# def get_resnet():
#     backbone = getattr(models, args.model)(weights=None, zero_init_residual=True)
#     h_dim = backbone.fc.in_features
#     backbone.fc = nn.Identity()
#     return backbone, h_dim


# TODO：加tokenizer对输入的原始字符串进行处理
class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        from uer.uer.encoders import str2encoder
        from uer.uer.embeddings import str2embedding
        self.embedding = str2embedding['bert'](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder['transformer'](args)
        self.fc = nn.Identity()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x)
        return x

def get_bert(args):
    bert = BERT(args)
    return bert, args.hidden_size

def mlp_mapper(h_dim, arch, bn_end=False, h_dim2=None):
    if h_dim2 is None:
        h_dim2 = h_dim
    arch = f"{h_dim}-{arch}-{h_dim2}"
    f = list(map(int, arch.split('-')))
    layers = []
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1]))
    if bn_end:
        layers[-1].bias.requires_grad = False
        layers.append(nn.BatchNorm1d(f[-1], affine=False))

    return nn.Sequential(*layers)

#############################################
#              Model Definition             #
#############################################

class SimSiam(nn.Module):

    def __init__(self, backbone, h_dim):
        super(SimSiam, self).__init__()
        self.backbone = backbone
        self.backbone.fc = mlp_mapper(h_dim, args.proj_arch, bn_end=True)
        self.predictor = mlp_mapper(h_dim, args.pred_arch)
        self.h_dim = h_dim
        return
    
    def forward(self, x1, x2):
        z1, z2 = self.backbone(x1), self.backbone(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

def make_net(args):
    # backbone, h_dim = get_resnet()
    # backbone, h_dim = get_bert(args)
    from encoder import get_longformer
    backbone, h_dim = get_longformer(args)
    model = SimSiam(backbone, h_dim)
    model.to(args.device)

    if is_distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    return model

##############################
#        Scheduling          #
##############################

def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    init_lr = args.base_lr * args.batch_size / 256
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    
    return cur_lr
    
###########################################
#          Logging & Checkpoints          #
###########################################

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


def resume(net, optimizer):
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        print("No checkpoint found!")
        return 0, float("inf")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch, best_loss = ckpt["epoch"]+1, ckpt["best_loss"]
    print(f"Checkpoint found! Resuming from epoch {epoch}")
    return epoch, best_loss

def save_checkpoint(state, is_best, filename="checkpoint.pt"):
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

########################################
#           Train and Eval             #
########################################

# class ImageFolderSubset(datasets.ImageFolder):
#     def __init__(self, subset_txt, **kwargs):
#         super().__init__(**kwargs)
#         if subset_txt is not None:
#             with open(subset_txt, 'r') as fid:
#                 subset = set([line.strip() for line in fid.readlines()])

#             subset_samples = []
#             for sample in self.samples:
#                 if os.path.basename(sample[0]) in subset:
#                     subset_samples.append(sample)

#             self.samples = subset_samples
#             self.targets = [s[1] for s in subset_samples]

@torch.no_grad()
def eval(net, args):
    def _load_data(args):
        from custom_datasets import CustomDataset
        from torch.utils.data import random_split
        from custom_datasets import collate_ContrastiveDataset_test

        dataset = CustomDataset(args.test_dir)
        classes = len(dataset.id2label)
        total_len = len(dataset)
        len_8 = int(total_len * 0.8)
        len_2 = total_len - len_8

        subset_8, subset_2 = random_split(dataset, [len_8, len_2], generator=torch.Generator().manual_seed(42))

        sampler_8 = torch.utils.data.distributed.DistributedSampler(subset_8) if is_distributed() else None
        sampler_2 = torch.utils.data.distributed.DistributedSampler(subset_2) if is_distributed() else None

        loader_8 = torch.utils.data.DataLoader(subset_8, batch_size=args.per_device_batch_size,
                                               num_workers=args.workers, pin_memory=True,
                                               drop_last=True, sampler=sampler_8, collate_fn=collate_ContrastiveDataset_test)
        loader_2 = torch.utils.data.DataLoader(subset_2, batch_size=args.per_device_batch_size,
                                               num_workers=args.workers, pin_memory=True,
                                               drop_last=True, sampler=sampler_2, collate_fn=collate_ContrastiveDataset_test)
        return loader_8, loader_2, classes

    def predict(features, X_train, Y_train, classes, n=200, t=0.2):
        scores = features @ X_train.t()
        weights, idx = scores.topk(k=n, dim=-1)
        labels = torch.gather(Y_train.expand(features.size(0), -1), dim=-1, index=idx)
        weights = (weights / t).exp()
        oh_labels = torch.zeros(features.size(0) * n, classes, device=args.device)
        oh_labels = oh_labels.scatter(dim=-1, index=labels.view(-1, 1), value=1.0)
        preds = torch.sum(oh_labels.view(features.size(0), -1, classes) * weights.unsqueeze(dim=-1), dim=1)
        return preds.argsort(dim=-1, descending=True)

    correct, total = 0, 0
    features_bank, labels_bank = [], []
    
    net.eval()
    backbone = net.module.backbone if hasattr(net, 'module') else net.backbone
    train_loader, test_loader, classes = _load_data(args)

    # Extract features from training set
    with torch.no_grad():
        for (x, mask_len, labels) in train_loader:
            attention_mask = torch.zeros_like(x)
            for i in range(x.size(0)):
                attention_mask[i, :mask_len[i]] = 1
            global_attention_mask = torch.zeros_like(x)
            global_attention_mask[:, 0] = 1
            x = (
                x.to(args.device, non_blocking=True),
                attention_mask.to(args.device, non_blocking=True),
                global_attention_mask.to(args.device, non_blocking=True),
            )
            labels = labels.to(args.device, non_blocking=True)
            features_bank.append(F.normalize(backbone(x), dim=-1))
            labels_bank.append(labels)
    
    features_bank = torch.cat(features_bank, dim=0)
    labels_bank = torch.cat(labels_bank, dim=0)
    
    # Synchronize before gathering
    if is_distributed():
        dist.barrier()

    # Gather features from all GPUs
    if is_distributed():
        features_bank_list = [torch.zeros_like(features_bank) for _ in range(dist.get_world_size())]
        labels_bank_list = [torch.zeros_like(labels_bank) for _ in range(dist.get_world_size())]
        
        # print(get_rank(), "features_bank.shape", features_bank.shape, "labels_bank.shape", labels_bank.shape)

        dist.all_gather(features_bank_list, features_bank)
        dist.all_gather(labels_bank_list, labels_bank)
        
        features_bank = torch.cat(features_bank_list, dim=0)
        labels_bank = torch.cat(labels_bank_list, dim=0)

    # classes = len(test_loader.dataset.classes)

    # Evaluate
    with torch.no_grad():
        for (x, mask_len, labels) in test_loader:
            attention_mask = torch.zeros_like(x)
            for i in range(x.size(0)):
                attention_mask[i, :mask_len[i]] = 1
            global_attention_mask = torch.zeros_like(x)
            global_attention_mask[:, 0] = 1
            x = (
                x.to(args.device, non_blocking=True),
                attention_mask.to(args.device, non_blocking=True),
                global_attention_mask.to(args.device, non_blocking=True),
            )
            labels = labels.to(args.device, non_blocking=True)
            features = F.normalize(backbone(x), dim=-1)
            preds = predict(features, features_bank, labels_bank, classes, args.knn_n, args.knn_t)
            total += labels.size(0)
            correct += (preds[:, 0] == labels).sum().item()

    if args.distributed:
        correct = torch.tensor(correct).to(args.device)
        total = torch.tensor(total).to(args.device)
        dist.all_reduce(correct)
        dist.all_reduce(total)
        correct = correct.item()
        total = total.item()

    net.train()
    return (correct / total) * 100

def main(args):
    init_distributed(args)

    if args.eval_mode:
        net = make_net(args)
        if args.resume:
            resume(net, None)
        correct = eval(net, args)
        print(f"KNN accuracy: {correct:.4f} %")
        return

    #############
    ##   INIT  ##
    #############
    if args.is_master:
        for k, v in vars(args).items():
            print(f"{k}: {v}")
    device = args.device
    start_epoch = 0
    best_loss = float("inf")
    loss_fn = nn.CosineSimilarity(dim=-1).to(device)
    net = make_net(args)
    # print(net)
    print(f"number of params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    if args.fix_pred_lr:
        model = net.module if is_distributed() else net
        optims_params = [{'params': model.backbone.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optims_params = net.parameters()

    #############
    ##   DATA  ##
    #############

    optimizer = torch.optim.SGD(optims_params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)    
    # dataset = datasets.ImageFolder(args.train_dir, augment())
    from custom_datasets import CustomDataset
    dataset = CustomDataset(args.train_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
    scaler = torch.amp.GradScaler()
    from custom_datasets import collate_ContrastiveDataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=(sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True, collate_fn=lambda batch: collate_ContrastiveDataset(batch, args))

    if args.test_mode:
        if args.is_master:
            net.train()
            state = dict(epoch=1, model=net.state_dict(), optimizer=optimizer.state_dict(), best_loss=best_loss)
            save_checkpoint(state, True)
        dist.barrier()
        return

    #############
    ##  TRAIN  ##
    #############

    if args.resume:
        start_epoch, best_loss = resume(net, optimizer)
    
    last_logging = time.time()
    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter("Loss", ":.4f")
        if is_distributed(): sampler.set_epoch(epoch)
        
        net.train()
        lr = adjust_learning_rate(optimizer, epoch)
        step_count_of_epoch = len(loader)
        for step, (x1, x1_mask_len, x2, x2_mask_len, label) in enumerate(loader, start=epoch*len(loader)):
            # attention_mask中，1的位置是[0, x_mask_len)
            attention_mask_1 = torch.zeros_like(x1)
            attention_mask_2 = torch.zeros_like(x2)
            # 假设x1_mask_len, x2_mask_len shape: [batch]
            for i in range(x1.size(0)):
                attention_mask_1[i, :x1_mask_len[i]] = 1
            for i in range(x2.size(0)):
                attention_mask_2[i, :x2_mask_len[i]] = 1
            global_attention_mask_1 = torch.zeros_like(x1)
            global_attention_mask_2 = torch.zeros_like(x2)
            global_attention_mask_1[:, 0] = 1
            global_attention_mask_2[:, 0] = 1
            # print(x1.shape, attention_mask_1.shape, global_attention_mask_1.shape)
            x1 = (
                x1.to(args.device, non_blocking=True),
                attention_mask_1.to(args.device, non_blocking=True),
                global_attention_mask_1.to(args.device, non_blocking=True),
            )
            x2 = (
                x2.to(args.device, non_blocking=True),
                attention_mask_2.to(args.device, non_blocking=True),
                global_attention_mask_2.to(args.device, non_blocking=True),
            )
            label = label.to(args.device, non_blocking=True)

            optimizer.zero_grad() 
            with torch.autocast(device_type=args.device.type):
                p1, p2, z1, z2 = net(x1, x2) # batch_size * hidden_size
                loss = - (loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) * 0.5 # batch_size
                loss = loss * label

            loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), label.size(0))
            if args.is_master and step % args.log_freq == 0:
                current_time = time.time()
                print(f"Epoch: {epoch}, Step: {step}/{step_count_of_epoch*(epoch+1)}, Loss: {losses.avg:.4f}, Time: {current_time - last_logging:.2f}s")
                last_logging = current_time
                import sys
                sys.stdout.flush()

        knn_acc = eval(net, args)
        if args.is_master:
            is_best = losses.avg < best_loss
            if is_best: best_loss = losses.avg
            if (epoch % args.save_freq == 0) or is_best:
                state = dict(epoch=epoch, model=net.state_dict(), optimizer=optimizer.state_dict(), best_loss=best_loss)
                save_checkpoint(state, is_best)
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=best_loss, lr=lr, knn_acc=knn_acc)
            print(json.dumps(logs))
            print(f"Epoch {epoch} => KNN accuracy ({knn_acc:.4f} %)")


    if is_distributed():
        dist.destroy_process_group()
    return


##############################
#      Hyperparameters       #
##############################

def add_simsiam_args(parser):
    # training
    parser.add_argument('--train_dir', type=str, default='/datasets/train/', help="训练数据目录")
    parser.add_argument('--test_dir', type=str, default='/datasets/test/', help="测试数据目录")
    parser.add_argument('--save_dir', type=str, default='./out/', help="模型保存目录")
    parser.add_argument('--log_freq', type=int, default=5, help="日志打印频率")
    parser.add_argument('--save_freq', type=int, default=10, help="模型保存频率（每多少个epoch保存一次）")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=256, help="总batch size（会被world_size整除）")
    parser.add_argument('--resume', action='store_true', default=False, help="是否从checkpoint恢复训练")
    parser.add_argument('--seed', type=int, default=None, help="随机种子")

    # eval
    parser.add_argument('--eval_mode', action='store_true', default=False, help="是否启用eval模式")
    parser.add_argument('--test_mode', action='store_true', default=False, help="是否启用test模式")
    parser.add_argument('--subset_dir', type=str, default='/datasets/1percent.txt', help="子集用于eval")
    parser.add_argument('--eval_dir', type=str, default='/datasets/val/', help="验证集目录")
    parser.add_argument('--knn_n', type=int, default=200, help="kNN中的邻居数量")
    parser.add_argument('--knn_t', type=float, default=0.2, help="kNN温度参数")

    # model
    parser.add_argument('--model', type=str, default='resnet50', help="主干模型名字")
    parser.add_argument('--proj_arch', type=str, default='2048-2048', help="投影头结构")
    parser.add_argument('--pred_arch', type=str, default='512', help="预测头结构")

    # optim
    parser.add_argument('--base_lr', type=float, default=0.05, help="基础学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help="动量")
    parser.add_argument('--wd', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--fix_pred_lr', action='store_true', default=True, help="预测头是否固定学习率")
    
    # hardware
    parser.add_argument('--workers', type=int, default=16, help="dataloader线程数")
    parser.add_argument('--device', type=str, default='cuda', help="设备类型(cpu, cuda, mps)")
    parser.add_argument('--no-distributed', dest='distributed', action='store_false', default=True, help="禁用分布式训练")

if __name__ == "__main__":
    from uer.uer.opts import training_opts, optimization_opts, model_opts, tokenizer_opts
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts(parser)
    # training_opts(parser)
    optimization_opts(parser)
    tokenizer_opts(parser)
    add_simsiam_args(parser)
    parser.add_argument("--weak_sample_rate", type=float, default=0.5)
    args = parser.parse_args()
    args.vocab_path = 'config/encryptd_vocab.txt'
    args.per_device_batch_size = args.batch_size

    main(args)