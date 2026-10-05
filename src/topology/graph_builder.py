"""
Vascular Topology Module: Arterial Graph Construction.

Converts 3D skeletonized vascular centerlines into mathematical geometric graphs
represented as PyTorch Geometric (PyG) Data objects with 8-dimensional node feature
vectors and 4-dimensional edge feature vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data

from src.topology.skeletonizer import VesselSkeletonizer


class ArterialGraphBuilder:
    """
    Constructs PyTorch Geometric arterial graphs from 3D skeletonized vessel masks.
    """

    NODE_DIM = 8
    EDGE_DIM = 4

    def __init__(
        self,
        skeletonizer: Optional[VesselSkeletonizer] = None,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        max_connection_distance: float = 1.75,
    ) -> None:
        """
        Args:
            skeletonizer: VesselSkeletonizer instance (or default if None).
            spacing: Physical voxel spacing in millimeters (dx, dy, dz).
            max_connection_distance: Maximum distance in voxels to consider two
                skeleton voxels adjacent (1.75 covers 26-connectivity).
        """
        self.skeletonizer = skeletonizer or VesselSkeletonizer()
        self.spacing = np.array(spacing, dtype=np.float32)
        self.max_connection_distance = max_connection_distance

    def build_graph_from_mask(
        self,
        vessel_mask: np.ndarray,
        intensity_volume: Optional[np.ndarray] = None,
    ) -> Data:
        """
        End-to-end pipeline: computes distance transform, extracts skeleton,
        and constructs the full PyTorch Geometric arterial graph.

        Args:
            vessel_mask: 3D binary array of vessel segmentation.
            intensity_volume: Optional normalized 3D intensity scan (MRA/CTA)
                to extract voxel intensity at skeleton nodes.

        Returns:
            torch_geometric.data.Data object with:
                - x: (N, 8) FloatTensor of node features
                - edge_index: (2, 2*E) LongTensor of bidirectional edge connections
                - edge_attr: (2*E, 4) FloatTensor of edge geometric features
                - pos: (N, 3) FloatTensor of physical coordinates in mm
        """
        mask_bool = vessel_mask > 0
        if not np.any(mask_bool):
            return self._empty_fallback_graph(vessel_mask.shape)

        # 1. Physical caliber via Distance Transform
        edt = self.skeletonizer.compute_distance_transform(mask_bool, spacing=tuple(self.spacing))

        # 2. Extract 3D centerline skeleton
        skeleton = self.skeletonizer.skeletonize(mask_bool)
        if not np.any(skeleton):
            return self._empty_fallback_graph(vessel_mask.shape)

        # 3. Build graph from skeleton and EDT
        return self.build_graph(
            skeleton=skeleton,
            distance_transform=edt,
            volume_shape=vessel_mask.shape,
            intensity_volume=intensity_volume,
        )

    def build_graph(
        self,
        skeleton: np.ndarray,
        distance_transform: np.ndarray,
        volume_shape: Tuple[int, int, int],
        intensity_volume: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Constructs the PyG graph from an existing skeleton and distance transform.
        """
        coords_voxel = np.argwhere(skeleton > 0)  # (N, 3)
        num_nodes = len(coords_voxel)
        if num_nodes == 0:
            return self._empty_fallback_graph(volume_shape)

        # Physical coordinates in millimeters: x * dx, y * dy, z * dz
        pos_phys = coords_voxel.astype(np.float32) * self.spacing

        # Neighbor counts for 26-connectivity
        neighbor_counts = self.skeletonizer.get_neighbor_counts(skeleton)
        node_degrees = neighbor_counts[coords_voxel[:, 0], coords_voxel[:, 1], coords_voxel[:, 2]]

        # Radii in millimeters from distance transform
        radii_mm = distance_transform[coords_voxel[:, 0], coords_voxel[:, 1], coords_voxel[:, 2]]

        # Optional voxel intensity
        if intensity_volume is not None:
            intensities = intensity_volume[coords_voxel[:, 0], coords_voxel[:, 1], coords_voxel[:, 2]].astype(np.float32)
        else:
            intensities = np.zeros(num_nodes, dtype=np.float32)

        # Fast edge connection via KDTree
        tree = KDTree(coords_voxel)
        # 26-connectivity distance in voxels is <= sqrt(3) ~ 1.732
        pairs = tree.query_pairs(r=self.max_connection_distance, output_type="ndarray")

        if len(pairs) == 0:
            # Graph with no edges (isolated nodes)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.EDGE_DIM), dtype=torch.float32)
        else:
            src = pairs[:, 0]
            dst = pairs[:, 1]

            # Make bidirectional: (u -> v) and (v -> u)
            edges_src = np.concatenate([src, dst])
            edges_dst = np.concatenate([dst, src])
            edge_index = torch.tensor(np.stack([edges_src, edges_dst], axis=0), dtype=torch.long)

            # Compute Edge Features: [length_mm, dx/len, dy/len, dz/len]
            p_src = pos_phys[edges_src]
            p_dst = pos_phys[edges_dst]
            diff = p_dst - p_src  # (2E, 3)
            lengths = np.linalg.norm(diff, axis=1, keepdims=True)  # (2E, 1)
            unit_dir = np.divide(diff, np.maximum(lengths, 1e-6))  # (2E, 3)
            edge_features = np.concatenate([lengths, unit_dir], axis=1)  # (2E, 4)
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        # Compute 8-dimensional Node Features:
        # 1-3: Normalized spatial coordinates in [0, 1]
        shape_arr = np.array(volume_shape, dtype=np.float32)
        norm_coords = coords_voxel.astype(np.float32) / np.maximum(shape_arr, 1.0)

        # 4: Local vessel radius in mm
        norm_radius = radii_mm[:, np.newaxis]

        # 5: Normalized degree (degree / 10.0)
        norm_degree = (node_degrees.astype(np.float32) / 10.0)[:, np.newaxis]

        # 6: Topological node type flag (0.0=endpoint, 0.5=waypoint, 1.0=bifurcation/junction)
        node_types = np.zeros((num_nodes, 1), dtype=np.float32)
        node_types[node_degrees == 1] = 0.0
        node_types[node_degrees == 2] = 0.5
        node_types[node_degrees >= 3] = 1.0

        # 7: Local curvature / tortuosity heuristic (ratio of radius to degree)
        tortuosity = (radii_mm / np.maximum(node_degrees.astype(np.float32), 1.0))[:, np.newaxis]

        # 8: Intensity or centrality feature
        feat_intensity = intensities[:, np.newaxis]

        # Concatenate into (N, 8) feature matrix
        x_features = np.concatenate(
            [norm_coords, norm_radius, norm_degree, node_types, tortuosity, feat_intensity],
            axis=1,
        )

        return Data(
            x=torch.tensor(x_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(pos_phys, dtype=torch.float32),
            num_nodes=num_nodes,
        )

    def _empty_fallback_graph(self, volume_shape: Tuple[int, int, int]) -> Data:
        """Creates a safe single-node dummy graph when no vessel skeleton is present."""
        dummy_x = torch.zeros((1, self.NODE_DIM), dtype=torch.float32)
        dummy_pos = torch.zeros((1, 3), dtype=torch.float32)
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_attr = torch.empty((0, self.EDGE_DIM), dtype=torch.float32)

        return Data(
            x=dummy_x,
            edge_index=empty_edges,
            edge_attr=empty_attr,
            pos=dummy_pos,
            num_nodes=1,
        )
