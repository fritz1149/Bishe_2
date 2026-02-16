def get_GH(type):
    funcs = {
        'deactivated': add,
        'gh': GH,
        'gh++': GH_plus,
    }
    return funcs[type]

def add(grads1, grads2):
    return grads1 + grads2

def _flatten(grads):
    import torch
    parts = [g.reshape(-1) for g in grads if g is not None]
    return torch.cat(parts, dim=0) if parts else torch.tensor([])

def _unflatten(flat, grads_ref):
    result = []
    offset = 0
    for g in grads_ref:
        if g is None:
            result.append(None)
        else:
            numel = g.numel()
            result.append(flat[offset:offset+numel].reshape(g.shape))
            offset += numel
    return result

def GH(grads1, grads2):
    import torch
    g1 = _flatten(grads1)
    g2 = _flatten(grads2)
    dot = torch.dot(g1, g2)
    if dot < 0:
        norm1_sq = g1.norm() ** 2 + 1e-8
        norm2_sq = g2.norm() ** 2 + 1e-8
        g = g1 + g2 - (dot / norm2_sq) * g2 - (dot / norm1_sq) * g1
    else:
        g = g1 + g2
    return _unflatten(g, grads1)

def GH_plus(grads1, grads2, lam=0.5):
    import torch, math
    g1 = _flatten(grads1)
    g2 = _flatten(grads2)
    dot = torch.dot(g1, g2)
    if dot < 0:
        norm1 = g1.norm()
        norm2 = g2.norm()
        cos_sim = (dot / (norm1 * norm2 + 1e-8)).clamp(-1.0, 1.0)
        phi = torch.acos(cos_sim) - math.pi / 2
        scale1 = 1 + 2 * torch.sin(lam * phi / 2)
        scale2 = 1 + 2 * torch.sin((lam - 1) * phi / 2)
        g = scale1 * g1 + scale2 * g2
    else:
        g = g1 + g2
    return _unflatten(g, grads1)