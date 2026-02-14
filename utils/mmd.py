import torch
import numpy as np


def get_MMD(type_, weight_type):
    funcs = {
        'mmd': MMD,
        'lmmd': LMMD,
    }
    if type_ == 'lmmd':
        from functools import partial
        return partial(funcs[type_], weight_type)
    return funcs[type_]


def _gaussian_kernel(src_features, tgt_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算多核高斯RBF核矩阵，返回 (kernels, n_s, n_t) 或 None（遇到NaN时）"""
    n_s = src_features.size(0)
    n_t = tgt_features.size(0)

    total = torch.cat([src_features, tgt_features], dim=0)          # (N_s+N_t, D)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        n_all = total.size(0)
        bandwidth = torch.sum(L2_distance.data) / (n_all ** 2 - n_all)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = sum(torch.exp(-L2_distance / bw) for bw in bandwidth_list)

    if torch.isnan(kernels).any():
        return None, n_s, n_t
    return kernels, n_s, n_t


def _compute_mmd_loss(SS, TT, ST, n_s, n_t, weight_ss=None, weight_tt=None, weight_st=None, norm=False):
    """根据核矩阵块和（可选的）权重矩阵计算归一化MMD loss"""
    if weight_ss is not None:
        SS = weight_ss * SS
        TT = weight_tt * TT
        ST = weight_st * ST

    if norm:
        loss_ss = torch.sum(SS) / (n_s * n_s) if n_s > 0 else 0.0
        loss_tt = torch.sum(TT) / (n_t * n_t) if n_t > 0 else 0.0
        loss_st = torch.sum(ST) / (n_s * n_t) if (n_s > 0 and n_t > 0) else 0.0
    else:
        loss_ss = torch.sum(SS)
        loss_tt = torch.sum(TT)
        loss_st = torch.sum(ST)
    
    return loss_ss + loss_tt - 2 * loss_st


def MMD(batch_s, batch_t, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    src_features = torch.cat([x[0] for x in batch_s], dim=0)
    tgt_features = torch.cat([x[0] for x in batch_t], dim=0)

    kernels, n_s, n_t = _gaussian_kernel(src_features, tgt_features, kernel_mul, kernel_num, fix_sigma)
    if kernels is None:
        return torch.tensor(0.0, device=src_features.device, requires_grad=True)

    SS = kernels[:n_s, :n_s]
    TT = kernels[n_s:, n_s:]
    ST = kernels[:n_s, n_s:]
    return _compute_mmd_loss(SS, TT, ST, n_s, n_t, norm=True)


def LMMD(weight_type, batch_s, batch_t, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    src_features = torch.cat([x[0] for x in batch_s], dim=0)   # (N_s, D)
    src_labels = torch.cat([x[1] for x in batch_s], dim=0)     # (N_s,) integer
    tgt_features = torch.cat([x[0] for x in batch_t], dim=0)   # (N_t, D)
    tgt_logits = torch.cat([x[1] for x in batch_t], dim=0)     # (N_t, C)

    n_s = src_features.size(0)
    n_t = tgt_features.size(0)
    class_num = tgt_logits.size(1)
    tgt_probs = torch.nn.functional.softmax(tgt_logits, dim=1)

    kernels, _, _ = _gaussian_kernel(src_features, tgt_features, kernel_mul, kernel_num, fix_sigma)
    if kernels is None:
        return torch.tensor(0.0, device=src_features.device, requires_grad=True)

    SS = kernels[:n_s, :n_s]
    TT = kernels[n_s:, n_s:]
    ST = kernels[:n_s, n_s:]

    # ===== 类条件权重 =====
    s_np = src_labels.cpu().data.numpy()
    s_onehot = np.eye(class_num)[s_np]                          # (N_s, C)
    s_sum = s_onehot.sum(axis=0, keepdims=True)
    s_sum[s_sum == 0] = 100
    s_onehot = s_onehot / s_sum

    t_pred_np = tgt_probs.cpu().data.max(1)[1].numpy()          # (N_t,) pseudo labels
    t_np = tgt_probs.cpu().data.numpy()                         # (N_t, C)
    t_sum = t_np.sum(axis=0, keepdims=True)
    t_sum[t_sum == 0] = 100
    t_np = t_np / t_sum

    # 只保留源域和目标域共享的类
    shared = list(set(s_np.tolist()) & set(t_pred_np.tolist()))
    mask_s = np.zeros((n_s, class_num))
    mask_t = np.zeros((n_t, class_num))
    if shared:
        mask_s[:, shared] = 1
        mask_t[:, shared] = 1
    s_onehot = s_onehot * mask_s
    t_np = t_np * mask_t

    weight_ss = (s_onehot @ s_onehot.T).astype('float32')
    weight_tt = (t_np @ t_np.T).astype('float32')
    weight_st = (s_onehot @ t_np.T).astype('float32')

    if shared:
        weight_ss /= len(shared)
        weight_tt /= len(shared)
        weight_st /= len(shared)

    dev = src_features.device
    weight_ss = torch.from_numpy(weight_ss).to(dev)
    weight_tt = torch.from_numpy(weight_tt).to(dev)
    weight_st = torch.from_numpy(weight_st).to(dev)

    return _compute_mmd_loss(SS, TT, ST, n_s, n_t, weight_ss, weight_tt, weight_st)