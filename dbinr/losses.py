
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    """Per-pixel squared error; optionally down-weights background pixels."""

    def __init__(self, alpha=1.0, bg_weight=1.0):
        super().__init__()
        self.alpha, self.bg_weight = alpha, bg_weight

    def forward(self, pack):
        pred, gt = pack["output"]["inr_output"], pack["gt"]
        if self.bg_weight == 1.0:
            return self.alpha * F.mse_loss(pred, gt)
        w = torch.where(pack["seg_int"].unsqueeze(-1) == 0, self.bg_weight, 1.0)
        return self.alpha * (w * (pred - gt) ** 2).sum() / w.sum()


class FocalLoss(nn.Module):
    """Multiclass focal loss."""

    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pack):
        logits = pack["output"]["segmentation_output"]      # (B,N,C)
        tgt = pack["seg_int"]                                # (B,N)
        logp = F.log_softmax(logits, dim=-1)
        logp_t = torch.gather(logp, -1, tgt.unsqueeze(-1)).squeeze(-1)
        return self.alpha * (-((1 - logp_t.exp()) ** self.gamma) * logp_t).mean()


class CompositeLoss(nn.Module):
    def __init__(self, **losses):
        super().__init__()
        self.losses = nn.ModuleDict(losses)

    def forward(self, pack):
        total, info = 0.0, {}
        for k, fn in self.losses.items():
            v = fn(pack)
            total = total + v
            info[k] = float(v.detach())
        return total, info
