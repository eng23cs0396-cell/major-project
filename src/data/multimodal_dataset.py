"""
Multimodal Dataset Module: Pairs 3D Volumetric Image Patches with Local
Vascular Arterial Subgraphs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.data.dataset import AneurysmPatchDataset
from src.topology.graph_builder import ArterialGraphBuilder
from src.topology.subgraph_extractor import VesselSubgraphExtractor


class MultimodalAneurysmDataset(Dataset):
    """
    Pairs 3D volumetric image patches from AneurysmPatchDataset with corresponding
    k-hop local arterial topology subgraphs.
    """

    def __init__(
        self,
        base_dataset: AneurysmPatchDataset,
        graph_cache_dir: str = "artifacts/graph_cache",
        k_hops: int = 3,
        max_nodes: int = 64,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.base_dataset = base_dataset
        self.graph_cache_dir = graph_cache_dir
        self.spacing = spacing
        self.builder = ArterialGraphBuilder(spacing=spacing)
        self.extractor = VesselSubgraphExtractor(k_hops=k_hops, max_nodes=max_nodes)

        os.makedirs(self.graph_cache_dir, exist_ok=True)
        self._graph_memory_cache: Dict[str, Data] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_or_build_scan_graph(self, scan_id: str) -> Data:
        """Retrieves scan graph from memory or disk cache, or builds it from vessel mask."""
        if scan_id in self._graph_memory_cache:
            return self._graph_memory_cache[scan_id]

        cache_path = os.path.join(self.graph_cache_dir, f"{scan_id}.pt")
        if os.path.exists(cache_path):
            graph = torch.load(cache_path, weights_only=False)
            self._graph_memory_cache[scan_id] = graph
            return graph

        # Build graph from vessel mask
        base = self.base_dataset.dataset if hasattr(self.base_dataset, "dataset") else self.base_dataset
        vessel_masks_dir = getattr(base, "vessel_masks_dir", "dataset(topAneu)/vessel_masks")
        vessel_path = os.path.join(vessel_masks_dir, f"{scan_id}.nii.gz")
        if os.path.exists(vessel_path):
            vessel_nii = nib.load(vessel_path)
            vessel_mask = np.asarray(vessel_nii.dataobj, dtype=np.uint8) > 0
            del vessel_nii
            graph = self.builder.build_graph_from_mask(vessel_mask)
            del vessel_mask
        else:
            graph = self.builder._empty_fallback_graph((64, 64, 64))

        # Save to cache
        torch.save(graph, cache_path)
        self._graph_memory_cache[scan_id] = graph
        return graph

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        scan_id = item["scan_id"]
        centroid_voxel = item.get("centroid_voxel")

        full_graph = self._get_or_build_scan_graph(scan_id)

        if centroid_voxel is not None:
            query_pos = tuple(float(v) * s for v, s in zip(centroid_voxel, self.spacing))
            subgraph = self.extractor.extract_subgraph_around_point(full_graph, query_pos=query_pos)
        else:
            subgraph = self.extractor._fallback_subgraph(full_graph)

        item["graph"] = subgraph
        return item


def multimodal_collate_fn(batch: List[dict]) -> dict:
    """
    Collates a list of multimodal samples into batched PyTorch tensors
    and a batched PyTorch Geometric graph.
    """
    patches = torch.stack([item["patch"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    # Support both 'offset_voxel' and 'offset'
    if "offset_voxel" in batch[0]:
        offsets = torch.stack([item["offset_voxel"] for item in batch], dim=0)
    else:
        offsets = torch.stack([item["offset"] for item in batch], dim=0)

    # Support both 'size_mm' and 'diameter_mm'
    if "size_mm" in batch[0]:
        diameters = torch.stack([item["size_mm"] for item in batch], dim=0)
    else:
        diameters = torch.stack([item["diameter_mm"] for item in batch], dim=0)

    graphs = Batch.from_data_list([item["graph"] for item in batch])
    scan_ids = [item.get("scan_id", "") for item in batch]

    return {
        "patch": patches,
        "label": labels,
        "offset": offsets,
        "offset_voxel": offsets,
        "diameter_mm": diameters,
        "size_mm": diameters,
        "graph": graphs,
        "scan_id": scan_ids,
    }
