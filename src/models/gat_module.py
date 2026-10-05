"""
Vascular Topology Modeling: Graph Attention Network (GATv2) Module.

Processes local arterial subgraphs using multi-layer Graph Attention Networks
(GATv2) with edge geometric features (length, unit tangent directions) to capture
arterial branching geometry, bifurcation complexity, and vessel caliber patterns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GATv2Layer(nn.Module):
    """
    A single GATv2 convolution block with edge feature integration,
    residual connection, layer normalization, and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            concat=concat,
            dropout=dropout,
            add_self_loops=True,
        )
        total_out = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(total_out)
        self.dropout = nn.Dropout(dropout)

        if in_channels != total_out:
            self.residual = nn.Linear(in_channels, total_out)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = self.residual(x)
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.dropout(out)
        out = self.norm(out + res)
        return out


class VascularGATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network that encodes 3D arterial subgraphs
    into a fixed-size topological embedding vector z_topo.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 128,
        edge_dim: int = 4,
        num_layers: int = 3,
        num_heads: int = 4,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            in_channels: Dimension of node feature vector (default: 8).
            hidden_channels: Hidden feature channels per attention head (default: 64).
            out_channels: Final GAT layer channels per head (default: 128).
            edge_dim: Dimension of edge geometric attributes (default: 4).
            num_layers: Number of GATv2 message-passing layers (default: 3).
            num_heads: Number of attention heads (default: 4).
            embedding_dim: Output graph embedding dimension (default: 256).
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.layers = nn.ModuleList()

        # Layer 1: in_channels -> hidden_channels * num_heads
        self.layers.append(
            GATv2Layer(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=True,
            )
        )
        curr_in = hidden_channels * num_heads

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATv2Layer(
                    in_channels=curr_in,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            curr_in = hidden_channels * num_heads

        # Final GAT layer: output representation
        self.layers.append(
            GATv2Layer(
                in_channels=curr_in,
                out_channels=out_channels,
                heads=1,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=False,
            )
        )

        # Global pooling outputs: mean (out_channels) + max (out_channels) = 2 * out_channels
        pooled_dim = 2 * out_channels
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Encodes a batched PyG graph into a batched graph embedding.

        Args:
            data: torch_geometric.data.Batch or Data object containing:
                - x: (N_total, in_channels)
                - edge_index: (2, E_total)
                - edge_attr: (E_total, edge_dim)
                - batch: (N_total,) batch assignment tensor

        Returns:
            torch.Tensor: (B, embedding_dim) topological graph embedding vector z_topo.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Ensure edge_attr has compatible dimension if empty
        if edge_attr is None or edge_attr.size(0) == 0:
            edge_attr = torch.zeros((edge_index.size(1), 4), dtype=torch.float32, device=x.device)

        # Multi-layer GATv2 message passing
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr=edge_attr)

        # Dual pooling: global mean pool + global max pool
        mean_pooled = global_mean_pool(h, batch)
        max_pooled = global_max_pool(h, batch)
        graph_repr = torch.cat([mean_pooled, max_pooled], dim=-1)

        # Project to target embedding dimension (e.g. 256)
        z_topo = self.proj(graph_repr)
        return z_topo
