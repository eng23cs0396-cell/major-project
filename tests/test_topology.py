"""
Unit tests for Phase 4: Vascular Topology Pipeline (Skeletonization & Graph Construction).
"""

from __future__ import annotations

import os
import pytest
import numpy as np
import torch
import nibabel as nib
from torch_geometric.data import Data

from src.topology.skeletonizer import VesselSkeletonizer
from src.topology.graph_builder import ArterialGraphBuilder
from src.topology.subgraph_extractor import VesselSubgraphExtractor


def create_synthetic_bifurcation(shape=(40, 40, 40)) -> np.ndarray:
    """
    Creates a synthetic 3D Y-shaped vascular bifurcation mask.
    Stem along z axis (x=20, y=20, z=5..25), branching into two arms (z=25..35).
    """
    mask = np.zeros(shape, dtype=np.uint8)

    # Main trunk
    for z in range(5, 25):
        mask[18:23, 18:23, z] = 1

    # Left branch
    for step in range(10):
        z = 25 + step
        x = 20 - step
        mask[x - 1 : x + 2, 19:22, z] = 1

    # Right branch
    for step in range(10):
        z = 25 + step
        x = 20 + step
        mask[x - 1 : x + 2, 19:22, z] = 1

    return mask


class TestVesselSkeletonizer:
    def test_synthetic_skeletonization(self):
        mask = create_synthetic_bifurcation()
        skeletonizer = VesselSkeletonizer(pruning_threshold=2)

        skel = skeletonizer.skeletonize(mask)
        assert skel.dtype == bool
        assert np.sum(skel) > 0
        assert np.sum(skel) < np.sum(mask), "Skeleton should be much thinner than full vessel"

        # Distance transform
        edt = skeletonizer.compute_distance_transform(mask, spacing=(1.0, 1.0, 1.0))
        assert edt.shape == mask.shape
        assert edt[20, 20, 15] > 1.0, "Center of trunk should have radius > 1.0 voxel"

        # Classify nodes
        node_classes = skeletonizer.classify_nodes(skel)
        assert len(node_classes["endpoints"]) >= 3, "Y-bifurcation should have at least 3 terminal ends"
        assert len(node_classes["waypoints"]) > 0, "Centerline should contain waypoints"
        assert len(node_classes["junctions"]) >= 1, "Y-bifurcation should contain at least 1 junction"

    def test_empty_mask_handling(self):
        empty = np.zeros((30, 30, 30), dtype=np.uint8)
        skeletonizer = VesselSkeletonizer()
        skel = skeletonizer.skeletonize(empty)
        assert not np.any(skel)

        edt = skeletonizer.compute_distance_transform(empty)
        assert np.all(edt == 0)


class TestArterialGraphBuilder:
    def test_build_from_synthetic(self):
        mask = create_synthetic_bifurcation()
        builder = ArterialGraphBuilder(spacing=(0.5, 0.5, 0.5))

        graph = builder.build_graph_from_mask(mask)

        assert isinstance(graph, Data)
        assert graph.num_nodes > 10
        assert graph.x.size(1) == ArterialGraphBuilder.NODE_DIM, f"Node features should have dim {ArterialGraphBuilder.NODE_DIM}"
        assert graph.edge_attr.size(1) == ArterialGraphBuilder.EDGE_DIM, f"Edge features should have dim {ArterialGraphBuilder.EDGE_DIM}"
        assert graph.edge_index.size(0) == 2, "Edge index must have shape (2, E)"
        assert graph.edge_index.size(1) > 0, "Connected skeleton must yield edges"
        assert graph.pos.size(1) == 3, "Positions must be 3D coordinates"

    def test_empty_mask_fallback(self):
        builder = ArterialGraphBuilder()
        empty = np.zeros((20, 20, 20), dtype=np.uint8)
        fallback = builder.build_graph_from_mask(empty)

        assert fallback.num_nodes == 1
        assert fallback.x.size(1) == ArterialGraphBuilder.NODE_DIM
        assert fallback.edge_index.size(1) == 0
        assert fallback.edge_attr.size(1) == ArterialGraphBuilder.EDGE_DIM


class TestVesselSubgraphExtractor:
    def test_extract_subgraph_synthetic(self):
        mask = create_synthetic_bifurcation()
        builder = ArterialGraphBuilder(spacing=(1.0, 1.0, 1.0))
        full_graph = builder.build_graph_from_mask(mask)

        extractor = VesselSubgraphExtractor(k_hops=3, max_nodes=20)
        # Query near bifurcation center (20, 20, 25)
        sub_graph = extractor.extract_subgraph_around_point(full_graph, query_pos=(20.0, 20.0, 25.0))

        assert isinstance(sub_graph, Data)
        assert 1 < sub_graph.num_nodes <= 20
        assert sub_graph.x.size(1) == ArterialGraphBuilder.NODE_DIM
        assert sub_graph.edge_index.size(0) == 2
        assert hasattr(sub_graph, "seed_idx")
        assert 0 <= int(sub_graph.seed_idx) < sub_graph.num_nodes

    def test_far_query_fallback(self):
        mask = create_synthetic_bifurcation()
        builder = ArterialGraphBuilder()
        full_graph = builder.build_graph_from_mask(mask)

        extractor = VesselSubgraphExtractor(k_hops=3, max_seed_distance_mm=5.0)
        # Query extremely far away
        sub_graph = extractor.extract_subgraph_around_point(full_graph, query_pos=(1000.0, 1000.0, 1000.0))

        assert sub_graph.num_nodes == 1
        assert sub_graph.edge_index.size(1) == 0


class TestRealPatientScanTopology:
    @pytest.mark.skipif(
        not os.path.exists("dataset(topAneu)/vessel_masks/topaneu_center1_mr_001.nii.gz"),
        reason="Real patient vessel mask not available locally",
    )
    def test_real_scan_pipeline(self):
        path = "dataset(topAneu)/vessel_masks/topaneu_center1_mr_001.nii.gz"
        img = nib.load(path)
        vessel_mask = np.asarray(img.dataobj) > 0

        builder = ArterialGraphBuilder(spacing=(0.5, 0.5, 0.5))
        graph = builder.build_graph_from_mask(vessel_mask)

        assert graph.num_nodes >= 500, f"Expected at least 500 skeleton nodes, got {graph.num_nodes}"
        assert graph.edge_index.size(1) >= 500, f"Expected at least 500 edges, got {graph.edge_index.size(1)}"
        assert graph.x.size(1) == 8
        assert graph.edge_attr.size(1) == 4

        # Extract 3-hop subgraph around center of mass
        center = np.mean(graph.pos.numpy(), axis=0)
        extractor = VesselSubgraphExtractor(k_hops=3, max_nodes=64)
        sub = extractor.extract_subgraph_around_point(graph, query_pos=center)

        assert sub.num_nodes <= 64
        assert sub.x.size(1) == 8
        assert sub.seed_idx < sub.num_nodes
