import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiTaskLoss(nn.Module):

    def __init__(self, w_mask: float = 1.0, w_vad: float = 0.5, w_kws: float = 0.5,
                 kws_weight: torch.Tensor = None, label_smoothing: float = 0.05):
        super().__init__()
        self.w_mask = w_mask
        self.w_vad  = w_vad
        self.w_kws  = w_kws
        self.label_smoothing = label_smoothing
        self.register_buffer('kws_weight', kws_weight)

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets:     Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask_pred, vad_pred, kws_pred, _ = predictions
        mask_target, vad_target, kws_target = targets

        mt = mask_target.unsqueeze(1)
        mse_loss = F.mse_loss(mask_pred, mt)
        bce_mask = F.binary_cross_entropy(mask_pred, (mt > 0.5).float())
        mask_loss = mse_loss + bce_mask

        vad_loss = F.binary_cross_entropy(
            vad_pred.squeeze(-1), vad_target
        )

        w = self.kws_weight.to(kws_pred.device) if self.kws_weight is not None else None
        nll = F.nll_loss(kws_pred, kws_target, weight=w)
        eps = self.label_smoothing
        if eps > 0:
            smooth = -kws_pred.mean(dim=1).mean()
            kws_loss = (1.0 - eps) * nll + eps * smooth
        else:
            kws_loss = nll

        total = (
            self.w_mask * mask_loss
            + self.w_vad  * vad_loss
            + self.w_kws  * kws_loss
        )

        breakdown = {
            "mask": mask_loss.item(),
            "vad":  vad_loss.item(),
            "kws":  kws_loss.item(),
        }
        return total, breakdown


def build_loss(cfg, kws_weight: torch.Tensor = None) -> MultiTaskLoss:
    return MultiTaskLoss(cfg.loss.w_mask, cfg.loss.w_vad, cfg.loss.w_kws,
                         kws_weight=kws_weight)
