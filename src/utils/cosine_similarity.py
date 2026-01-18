
import torch

# x1: [B, C, H1, W1]
# x2: [B, C, H2, W2]
# return [B, H1, W1, H2, W2]
def cosine_similarity_BCHW(x1, x2, clamp=False):
    eps = 1e-8
    bsz, ch, ha, wa = x1.shape
    x1 = x1.view(bsz, ch, -1)
    x1 = x1 / (x1.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, hb, wb = x2.shape
    x2 = x2.view(bsz, ch, -1)
    x2 = x2 / (x2.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(x1.transpose(1, 2), x2).view(bsz, ha, wa, hb, wb)
    if clamp:
        corr = torch.clamp(corr)
    return corr


# x1: [B, N, C]
# x2: [B, K, C]
def cosine_similarity_BNC(x1, x2):
    eps = 1e-8
    x1 = x1 / (x1.norm(dim=2, p=2, keepdim=True) + eps)
    x2 = x2 / (x2.norm(dim=2, p=2, keepdim=True) + eps)

    corr = torch.bmm(x1, x2.transpose(1, 2))

    return corr

# x1: [N, C]
# x2: [K, C]
def cosine_similarity_NC(x1, x2):
    eps = 1e-8
    x1 = x1 / (x1.norm(dim=1, p=2, keepdim=True) + eps)
    x2 = x2 / (x2.norm(dim=1, p=2, keepdim=True) + eps)

    corr = x1 @ x2.t()

    return corr

# x1: [B, N, C]
# x2: [B, K, C]
def multi_head_cosine_similarity_BNC(x1, x2, head_dim):
    eps = 1e-8
    bsz, k, ch = x1.shape
    x1 = x1.permute(0, 2, 1).reshape(bsz,  ch // head_dim, head_dim, k).reshape(-1, head_dim, k)
    x1 = x1 / (x1.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, n, ch = x2.shape
    x2 = x2.permute(0, 2, 1).reshape(bsz, ch // head_dim, head_dim, n).reshape(-1, head_dim, n)
    x2 = x2 / (x2.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(x1.transpose(1, 2), x2).view(bsz, ch // head_dim, k, n)
    return corr


def multi_head_cosine_similarity_BCHW(query_feats, support_feats, head_dim):
    eps = 1e-5
    bsz, ch, hb, wb = support_feats.size()
    support_feats = support_feats.view(bsz, ch//head_dim, head_dim, -1).reshape(bsz * (ch//head_dim), head_dim, -1)
    support_feats = support_feats / (support_feats.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, ha, wa = query_feats.size()
    query_feats = query_feats.reshape(bsz, ch//head_dim, -1).reshape(bsz * (ch//head_dim), head_dim, -1)
    query_feats = query_feats / (query_feats.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(query_feats.transpose(1, 2), support_feats).view(bsz, ch//head_dim, ha, wa, hb, wb)
    corr = corr.clamp(min=0)

    return corr
