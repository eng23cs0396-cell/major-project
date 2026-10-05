"""
Multimodal Dual-Branch Detector: 3D CNN + Graph Attention Network (GATv2) with
Cross-Attention Fusion.

Integrates volumetric visual patch embeddings (Branch A) with local vascular
topological graph embeddings (Branch B) through multi-head cross-attention.
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.models.gat_module import VascularGATEncoder
from src.models.resnet3d import ResNet3DEncoder


class CrossAttentionFusionBlock(nn.Module):
    """
    Multi-Head Cross-Attention Fusion module.
    Queries: Visual features z_vis (from 3D ResNet)
    Keys/Values: Topological features z_topo (from GATv2)

    Allows the model to dynamically attend to vascular branching complexity
    and caliber changes that corroborate or refute the visual suspicion of an aneurysm.
    """

    def __init__(
        self,
        vis_dim: int = 512,
        topo_dim: int = 256,
        fused_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vis_dim = vis_dim
        self.topo_dim = topo_dim
        self.fused_dim = fused_dim

        # Projections to common cross-attention dimension
        self.q_proj = nn.Linear(vis_dim, fused_dim)
        self.k_proj = nn.Linear(topo_dim, fused_dim)
        self.v_proj = nn.Linear(topo_dim, fused_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(fused_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
        )
        self.norm2 = nn.LayerNorm(fused_dim)
        self.dropout2 = nn.Dropout(dropout)

        if vis_dim != fused_dim:
            self.res_proj = nn.Linear(vis_dim, fused_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, z_vis: torch.Tensor, z_topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_vis: (B, vis_dim) visual embedding from 3D CNN.
            z_topo: (B, topo_dim) topological embedding from GAT.

        Returns:
            torch.Tensor: (B, fused_dim) multimodal fused representation.
        """
        # Expand to sequence length 1: (B, 1, dim)
        q = self.q_proj(z_vis).unsqueeze(1)
        k = self.k_proj(z_topo).unsqueeze(1)
        v = self.v_proj(z_topo).unsqueeze(1)

        # Cross-Attention: Q from vision, K/V from topology
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = self.dropout1(attn_out.squeeze(1))

        # Add & Norm with residual from visual query
        res = self.res_proj(z_vis)
        x = self.norm1(res + attn_out)

        # FFN with residual connection
        ffn_out = self.dropout2(self.ffn(x))
        fused = self.norm2(x + ffn_out)
        return fused


class DualBranchDetector(nn.Module):
    """
    End-to-end multimodal 3D CNN + GATv2 architecture with Cross-Attention Fusion
    and multi-task prediction heads.
    """

    def __init__(
        self,
        backbone: Literal["resnet18_3d", "resnet34_3d"] = "resnet18_3d",
        in_channels: int = 1,
        vis_dim: int = 512,
        topo_in_channels: int = 8,
        topo_hidden_channels: int = 64,
        topo_out_channels: int = 128,
        topo_edge_dim: int = 4,
        topo_dim: int = 256,
        fused_dim: int = 512,
        dropout: float = 0.2,
        pretrained_baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Branch A: 3D Volumetric CNN Encoder
        layer_cfg = {
            "resnet18_3d": [2, 2, 2, 2],
            "resnet34_3d": [3, 4, 6, 3],
        }
        if backbone not in layer_cfg:
            raise ValueError(f"Unsupported backbone '{backbone}'.")

        self.visual_encoder = ResNet3DEncoder(layer_cfg[backbone], in_channels=in_channels)
        if vis_dim != self.visual_encoder.out_channels:
            self.visual_proj = nn.Linear(self.visual_encoder.out_channels, vis_dim)
        else:
            self.visual_proj = nn.Identity()

        # Branch B: Arterial Topology GATv2 Encoder
        self.topo_encoder = VascularGATEncoder(
            in_channels=topo_in_channels,
            hidden_channels=topo_hidden_channels,
            out_channels=topo_out_channels,
            edge_dim=topo_edge_dim,
            embedding_dim=topo_dim,
            dropout=dropout,
        )

        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusionBlock(
            vis_dim=vis_dim,
            topo_dim=topo_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # Multi-Task Prediction Heads
        self.cls_head = nn.Linear(fused_dim, 1)
        self.offset_head = nn.Linear(fused_dim, 3)
        self.size_head = nn.Linear(fused_dim, 1)

        # Optionally load pretrained visual baseline weights
        if pretrained_baseline_path is not None and os.path.exists(pretrained_baseline_path):
            self.load_visual_baseline_weights(pretrained_baseline_path)

    def load_visual_baseline_weights(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Loads weights from a Phase 3 baseline checkpoint into Branch A."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter visual encoder weights
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder.") :]] = v

        if encoder_state:
            missing, unexpected = self.visual_encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded visual encoder weights from {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    def forward(
        self,
        patches: torch.Tensor,
        graphs: Batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            patches: (B, 1, 64, 64, 64) 3D volumetric image patches.
            graphs: PyG Batch containing local vascular subgraphs.

        Returns:
            Dict containing:
                - aneurysm_logit: (B,)
                - aneurysm_prob: (B,)
                - offset: (B, 3)
                - diameter_mm: (B,)
                - z_vis: (B, vis_dim)
                - z_topo: (B, topo_dim)
                - z_fused: (B, fused_dim)
        """
        # Branch A: Visual feature extraction
        z_vis = self.visual_proj(self.visual_encoder(patches))

        # Branch B: Topological graph feature extraction
        z_topo = self.topo_encoder(graphs)

        # Multimodal Cross-Attention Fusion
        z_fused = self.fusion(z_vis, z_topo)

        # Multi-task heads
        logit = self.cls_head(z_fused).squeeze(-1)
        offset = self.offset_head(z_fused)
        diameter = F.relu(self.size_head(z_fused)).squeeze(-1)

        return {
            "aneurysm_logit": logit,
            "aneurysm_prob": torch.sigmoid(logit),
            "offset": offset,
            "diameter_mm": diameter,
            "z_vis": z_vis,
            "z_topo": z_topo,
            "z_fused": z_fused,
        }
