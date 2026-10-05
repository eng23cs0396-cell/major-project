"""
Vascular Topology Module: Local Arterial Subgraph Extractor.

Extracts k-hop local arterial subgraphs centered around 3D aneurysm candidate
locations or patch centroids. Prepares bounded subgraphs for the Graph Attention
Network (GAT) to fuse with 3D CNN volumetric patch embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph


class VesselSubgraphExtractor:
    """
    Extracts local k-hop arterial subgraphs from full cerebrovascular graphs.
    """

    def __init__(
        self,
        k_hops: int = 3,
        max_nodes: int = 64,
        max_seed_distance_mm: float = 30.0,
    ) -> None:
        """
        Args:
            k_hops: Number of edge traversals from the seed node (covers ~20-30mm).
            max_nodes: Upper bound on the number of nodes in the extracted subgraph.
            max_seed_distance_mm: Maximum allowable physical distance from query point
                to the nearest skeleton node before returning fallback graph.
        """
        self.k_hops = k_hops
        self.max_nodes = max_nodes
        self.max_seed_distance_mm = max_seed_distance_mm

    def extract_subgraph_around_point(
        self,
        full_graph: Data,
        query_pos: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
        tree: Optional[KDTree] = None,
    ) -> Data:
        """
        Locates the nearest vessel node to query_pos and extracts a k-hop subgraph.

        Args:
            full_graph: PyG Data object of the whole cerebral arterial tree.
            query_pos: (3,) coordinate in the same physical space as full_graph.pos.
            tree: Optional pre-built KDTree on full_graph.pos for fast querying.

        Returns:
            PyG Data object representing the local arterial subgraph with:
                - x: (M, 8) where M <= max_nodes
                - edge_index: (2, E_sub) relabeled local edge indices
                - edge_attr: (E_sub, 4) local edge features
                - pos: (M, 3) local node coordinates
                - seed_idx: index of the query center node within this subgraph
        """
        if full_graph.num_nodes <= 1 or full_graph.edge_index.size(1) == 0:
            return self._fallback_subgraph(full_graph)

        pos_np = full_graph.pos.cpu().numpy()
        query_np = np.asarray(query_pos, dtype=np.float32).ravel()

        if tree is None:
            tree = KDTree(pos_np)

        dist, seed_idx = tree.query(query_np)
        seed_idx = int(seed_idx)

        # If query point is too far from any vessel, return fallback
        if dist > self.max_seed_distance_mm:
            return self._fallback_subgraph(full_graph)

        # Extract k-hop neighborhood
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=seed_idx,
            num_hops=self.k_hops,
            edge_index=full_graph.edge_index,
            relabel_nodes=True,
        )

        # If subgraph exceeds max_nodes, retain the closest nodes to the seed
        if len(subset) > self.max_nodes:
            sub_pos = pos_np[subset.cpu().numpy()]
            distances = np.linalg.norm(sub_pos - query_np, axis=1)
            keep_indices = np.argsort(distances)[: self.max_nodes]

            # Re-index subset
            trimmed_subset = subset[keep_indices]
            sub_edge_index, sub_edge_attr = subgraph(
                subset=trimmed_subset,
                edge_index=full_graph.edge_index,
                edge_attr=full_graph.edge_attr,
                relabel_nodes=True,
            )
            final_subset = trimmed_subset
            # Find new index of seed
            seed_pos_in_trim = np.where(final_subset.cpu().numpy() == seed_idx)[0]
            mapped_seed = int(seed_pos_in_trim[0]) if len(seed_pos_in_trim) > 0 else 0
        else:
            final_subset = subset
            mapped_seed = int(mapping.item()) if hasattr(mapping, "item") else int(mapping[0])
            sub_edge_attr = full_graph.edge_attr[edge_mask] if full_graph.edge_attr is not None else None
        sub_x = full_graph.x[final_subset]
        sub_pos_tensor = full_graph.pos[final_subset]

        sub_data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            pos=sub_pos_tensor,
            num_nodes=len(final_subset),
        )
        sub_data.seed_idx = torch.tensor(mapped_seed, dtype=torch.long)
        return sub_data

    def _fallback_subgraph(self, reference_graph: Optional[Data] = None) -> Data:
        """Constructs a deterministic single-node dummy graph for empty or vessel-free regions."""
        node_dim = reference_graph.x.size(1) if reference_graph is not None and reference_graph.x is not None else 8
        edge_dim = (
            reference_graph.edge_attr.size(1)
            if reference_graph is not None and reference_graph.edge_attr is not None and reference_graph.edge_attr.size(0) > 0
            else 4
        )

        dummy = Data(
            x=torch.zeros((1, node_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim), dtype=torch.float32),
            pos=torch.zeros((1, 3), dtype=torch.float32),
            num_nodes=1,
        )
        dummy.seed_idx = torch.tensor(0, dtype=torch.long)
        return dummy
