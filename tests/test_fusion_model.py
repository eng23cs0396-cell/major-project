"""
Unit tests for Phase 5: GAT & Multimodal Cross-Attention Fusion Architecture.
"""

from __future__ import annotations

import os
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models.gat_module import VascularGATEncoder
from src.models.dual_branch_detector import CrossAttentionFusionBlock, DualBranchDetector
from src.models.losses import MultiTaskDetectionLoss
from src.data.multimodal_dataset import multimodal_collate_fn


def create_dummy_subgraph(num_nodes: int = 12, in_dim: int = 8, edge_dim: int = 4) -> Data:
    """Helper creating a synthetic PyG arterial subgraph."""
    x = torch.randn(num_nodes, in_dim)
    # Simple line / cycle graph
    src = torch.arange(num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    num_edges = edge_index.size(1)
    edge_attr = torch.randn(num_edges, edge_dim)
    pos = torch.randn(num_nodes, 3)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, num_nodes=num_nodes)


class TestVascularGATEncoder:
    def test_forward_and_backward(self):
        encoder = VascularGATEncoder(
            in_channels=8,
            hidden_channels=32,
            out_channels=64,
            edge_dim=4,
            num_layers=2,
            num_heads=2,
            embedding_dim=256,
        )

        g1 = create_dummy_subgraph(num_nodes=10)
        g2 = create_dummy_subgraph(num_nodes=15)
        batch = Batch.from_data_list([g1, g2])

        z_topo = encoder(batch)
        assert z_topo.shape == (2, 256), f"Expected shape (2, 256), got {z_topo.shape}"

        loss = z_topo.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"


class TestCrossAttentionFusion:
    def test_fusion_block(self):
        fusion = CrossAttentionFusionBlock(vis_dim=512, topo_dim=256, fused_dim=512, num_heads=4)
        z_vis = torch.randn(4, 512)
        z_topo = torch.randn(4, 256)

        z_fused = fusion(z_vis, z_topo)
        assert z_fused.shape == (4, 512), f"Expected fused shape (4, 512), got {z_fused.shape}"

        loss = z_fused.sum()
        loss.backward()
        assert fusion.q_proj.weight.grad is not None
        assert fusion.k_proj.weight.grad is not None


class TestDualBranchDetector:
    def test_forward_cpu(self):
        model = DualBranchDetector(
            backbone="resnet18_3d",
            in_channels=1,
            vis_dim=512,
            topo_in_channels=8,
            topo_hidden_channels=32,
            topo_out_channels=64,
            topo_dim=256,
            fused_dim=512,
        )
        model.eval()

        B = 2
        patches = torch.randn(B, 1, 64, 64, 64)
        g1 = create_dummy_subgraph(num_nodes=8)
        g2 = create_dummy_subgraph(num_nodes=12)
        graphs = Batch.from_data_list([g1, g2])

        with torch.no_grad():
            out = model(patches, graphs)

        assert "aneurysm_logit" in out
        assert "aneurysm_prob" in out
        assert "offset" in out
        assert "diameter_mm" in out
        assert out["aneurysm_logit"].shape == (B,)
        assert out["aneurysm_prob"].shape == (B,)
        assert out["offset"].shape == (B, 3)
        assert out["diameter_mm"].shape == (B,)
        assert torch.all(out["aneurysm_prob"] >= 0.0) and torch.all(out["aneurysm_prob"] <= 1.0)
        assert torch.all(out["diameter_mm"] >= 0.0)

    def test_training_step_with_loss(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualBranchDetector(
            backbone="resnet18_3d",
            topo_hidden_channels=32,
            topo_out_channels=64,
        ).to(device)
        model.train()

        criterion = MultiTaskDetectionLoss(lambda_offset=0.1, lambda_size=0.1, focal_alpha=0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        B = 2
        patches = torch.randn(B, 1, 64, 64, 64, device=device)
        g1 = create_dummy_subgraph(num_nodes=6)
        g2 = create_dummy_subgraph(num_nodes=10)
        graphs = Batch.from_data_list([g1, g2]).to(device)

        labels = torch.tensor([1, 0], dtype=torch.long, device=device)
        offsets = torch.randn(B, 3, device=device)
        diameters = torch.tensor([4.5, 0.0], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        outputs = model(patches, graphs)
        losses = criterion(outputs, labels, offsets, diameters)

        assert losses["loss"].item() > 0.0
        losses["loss"].backward()
        optimizer.step()


class TestMultimodalCollate:
    def test_collate_fn(self):
        sample1 = {
            "patch": torch.randn(1, 64, 64, 64),
            "label": torch.tensor(1, dtype=torch.long),
            "offset": torch.tensor([0.5, -0.2, 0.1]),
            "diameter_mm": torch.tensor(3.8),
            "graph": create_dummy_subgraph(num_nodes=5),
            "scan_id": "scan_001",
        }
        sample2 = {
            "patch": torch.randn(1, 64, 64, 64),
            "label": torch.tensor(0, dtype=torch.long),
            "offset": torch.tensor([0.0, 0.0, 0.0]),
            "diameter_mm": torch.tensor(0.0),
            "graph": create_dummy_subgraph(num_nodes=7),
            "scan_id": "scan_002",
        }

        batch = multimodal_collate_fn([sample1, sample2])
        assert batch["patch"].shape == (2, 1, 64, 64, 64)
        assert batch["label"].shape == (2,)
        assert batch["offset"].shape == (2, 3)
        assert batch["diameter_mm"].shape == (2,)
        assert isinstance(batch["graph"], Batch)
        assert batch["graph"].num_graphs == 2
