"""
Multi-task losses for the image-only 3D aneurysm baseline.

L_total = L_focal + lambda_offset * L_offset + lambda_size * L_size
Offset and diameter terms are computed on positive (aneurysm) samples only.
Focal alpha defaults to 0.75 so aneurysm patches are not under-weighted
when positives and negatives are sampled evenly.
Regression weights default below 1 so diameter/offset do not drown classification.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskDetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_offset: float = 0.1,
        lambda_size: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        offset_target: torch.Tensor,
        size_target: torch.Tensor,
        positive_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.float()
        if positive_mask is None:
            positive_mask = labels > 0.5

        cls_loss = self.focal(outputs["aneurysm_logit"], labels)

        pos = positive_mask.bool()
        if pos.any():
            offset_loss = self.huber(outputs["offset"][pos], offset_target[pos])
            size_loss = self.huber(outputs["diameter_mm"][pos], size_target[pos])
        else:
            offset_loss = outputs["offset"].sum() * 0.0
            size_loss = outputs["diameter_mm"].sum() * 0.0

        total = cls_loss + self.lambda_offset * offset_loss + self.lambda_size * size_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "offset_loss": offset_loss.detach(),
            "size_loss": size_loss.detach(),
        }
