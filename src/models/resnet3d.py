"""
3D ResNet baseline for 64^3 angiographic patches.

Produces a 512-d visual embedding and three task heads:
aneurysm probability, voxel centroid offset, and physical diameter (mm).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class ResNet3DEncoder(nn.Module):
    def __init__(self, layers, in_channels: int = 1) -> None:
        super().__init__()
        self.in_planes = 64
        # Keep resolution on the first conv so 64^3 medical patches retain small-lesion detail.
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.out_channels = 512

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * BasicBlock3D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet3DBaseline(nn.Module):
    """Image-only multi-task detector used as the Phase 3 visual baseline."""

    def __init__(
        self,
        backbone: Literal["resnet18_3d", "resnet34_3d"] = "resnet18_3d",
        in_channels: int = 1,
        embedding_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layer_cfg = {
            "resnet18_3d": [2, 2, 2, 2],
            "resnet34_3d": [3, 4, 6, 3],
        }
        if backbone not in layer_cfg:
            raise ValueError(f"Unsupported backbone '{backbone}'. Expected resnet18_3d or resnet34_3d.")

        self.encoder = ResNet3DEncoder(layer_cfg[backbone], in_channels=in_channels)
        self.embedding_dim = embedding_dim
        if embedding_dim != self.encoder.out_channels:
            self.proj = nn.Linear(self.encoder.out_channels, embedding_dim)
        else:
            self.proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        hidden = embedding_dim
        self.cls_head = nn.Linear(hidden, 1)
        self.offset_head = nn.Linear(hidden, 3)
        self.size_head = nn.Linear(hidden, 1)

    def forward(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_vis = self.proj(self.encoder(patches))
        z = self.dropout(z_vis)
        logit = self.cls_head(z).squeeze(-1)
        offset = self.offset_head(z)
        diameter = F.relu(self.size_head(z)).squeeze(-1)
        return {
            "embedding": z_vis,
            "aneurysm_logit": logit,
            "aneurysm_prob": torch.sigmoid(logit),
            "offset": offset,
            "diameter_mm": diameter,
        }


def build_baseline_model(
    backbone: str = "resnet18_3d",
    in_channels: int = 1,
    embedding_dim: int = 512,
    dropout: float = 0.2,
) -> ResNet3DBaseline:
    return ResNet3DBaseline(
        backbone=backbone,  # type: ignore[arg-type]
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


def count_parameters(model: Optional[nn.Module] = None) -> int:
    model = model or build_baseline_model()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
