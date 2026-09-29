"""
PyTorch Volumetric Dataset for Intracranial Aneurysm Detection & Patch Training.
Supports balanced sampling between true aneurysm sites (positive) and
normal vessel bifurcation/branching locations (hard negative).

Includes:
- In-memory patch caching for ultra-fast epoch iteration.
- Deterministic negative patch evaluation for reproducible validation.
- Anatomic safety margin preventing false-negative supervision on diseased scans.
- 3D spatial flip data augmentation with synchronized offset transformation.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessor import VolumetricPreprocessor


class AneurysmPatchDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        eda_summary_csv: Optional[str] = "artifacts/eda/aneurysm_annotations_summary.csv",
        dataset_root: str = "dataset(topAneu)",
        preprocessor: Optional[VolumetricPreprocessor] = None,
        neg_pos_ratio: float = 2.0,  # 2 negative vessel patches per positive patch
        jitter_range: int = 2,
        is_training: bool = True,
        cache_patches: bool = True,
        min_neg_distance_voxels: float = 20.0,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = preprocessor or VolumetricPreprocessor()
        self.jitter_range = jitter_range if is_training else 0
        self.is_training = is_training
        self.cache_patches = cache_patches
        self.min_neg_distance_voxels = min_neg_distance_voxels

        self.images_dir = os.path.join(dataset_root, "images")
        self.loc_masks_dir = os.path.join(dataset_root, "location_masks")
        self.vessel_masks_dir = os.path.join(dataset_root, "vessel_masks")
        self.loc_jsons_dir = os.path.join(dataset_root, "location_jsons")

        # Load split scans
        self.split_df = pd.read_csv(split_csv)
        self.scan_ids = set(self.split_df["scan_id"])

        # Track known aneurysm centroids per scan to guard against negative contamination
        self.aneurysms_by_scan: Dict[str, List[Tuple[int, int, int]]] = {}

        # Load aneurysm metadata if available
        self.samples: List[Dict] = []
        if eda_summary_csv and os.path.exists(eda_summary_csv):
            eda_df = pd.read_csv(eda_summary_csv)
            split_aneurysms = eda_df[eda_df["scan_id"].isin(self.scan_ids)]
            for _, row in split_aneurysms.iterrows():
                centroid = (
                    int(row["centroid_voxel_x"]),
                    int(row["centroid_voxel_y"]),
                    int(row["centroid_voxel_z"]),
                )
                scan_id = row["scan_id"]
                if scan_id not in self.aneurysms_by_scan:
                    self.aneurysms_by_scan[scan_id] = []
                self.aneurysms_by_scan[scan_id].append(centroid)

                self.samples.append(
                    {
                        "scan_id": scan_id,
                        "modality": row["modality"],
                        "center": row["center"],
                        "is_aneurysm": 1,
                        "centroid_voxel": centroid,
                        "max_diameter_mm": float(row["max_diameter_mm"]),
                        "size_category": row["size_category"],
                        "location_name": row["location_name"],
                    }
                )

        # Pre-assign negative candidate slots
        num_positives = max(1, len(self.samples))
        target_negatives = int(num_positives * neg_pos_ratio)
        scans_for_negs = self.split_df["scan_id"].values
        rng = np.random.RandomState(42)

        for i in range(target_negatives):
            s_id = rng.choice(scans_for_negs)
            row = self.split_df[self.split_df["scan_id"] == s_id].iloc[0]
            self.samples.append(
                {
                    "scan_id": s_id,
                    "modality": row["modality"],
                    "center": row["center"],
                    "is_aneurysm": 0,
                    "centroid_voxel": None,  # sampled on first access and cached
                    "sample_seed": int(rng.randint(0, 1_000_000)),
                    "max_diameter_mm": 0.0,
                    "size_category": "Normal Vessel",
                    "location_name": "Normal Branch",
                }
            )

        # In-memory caches for fast loading
        self._vessel_pts_cache: Dict[str, np.ndarray] = {}
        self._patch_cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]] = {}

        print(
            f"Dataset initialized ({'Train' if is_training else 'Val/Test'}): "
            f"{len(self.samples)} total candidate patches "
            f"({num_positives} positive, {len(self.samples) - num_positives} negative) | "
            f"Cache enabled: {self.cache_patches}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_vessel_points(self, scan_id: str) -> np.ndarray:
        if scan_id in self._vessel_pts_cache:
            return self._vessel_pts_cache[scan_id]

        vessel_path = os.path.join(self.vessel_masks_dir, f"{scan_id}.nii.gz")
        if os.path.exists(vessel_path):
            vessel_data = nib.load(vessel_path).get_fdata()
            pts = np.argwhere(vessel_data > 0)
            # Safety guard: filter out vessel points within min distance of known aneurysms
            if scan_id in self.aneurysms_by_scan and len(pts) > 0:
                for ac in self.aneurysms_by_scan[scan_id]:
                    dists = np.linalg.norm(pts - np.array(ac), axis=1)
                    pts = pts[dists >= self.min_neg_distance_voxels]
            self._vessel_pts_cache[scan_id] = pts
        else:
            self._vessel_pts_cache[scan_id] = np.empty((0, 3), dtype=np.int32)

        return self._vessel_pts_cache[scan_id]

    def _get_negative_centroid(self, scan_id: str, vol_shape: Tuple[int, ...], seed: int) -> Tuple[int, int, int]:
        pts = self._get_vessel_points(scan_id)
        if len(pts) > 0:
            local_rng = np.random.RandomState(seed)
            choice = local_rng.choice(len(pts))
            return tuple(int(v) for v in pts[choice])
        return tuple(int(v) for v in (np.array(vol_shape) // 2))

    def _load_base_patch(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]:
        if self.cache_patches and idx in self._patch_cache:
            return self._patch_cache[idx]

        sample = self.samples[idx]
        scan_id = sample["scan_id"]
        modality = sample["modality"]

        # Load full 3D scan
        img_path = os.path.join(self.images_dir, f"{scan_id}_0000.nii.gz")
        img_nii = nib.load(img_path)
        vol_data = img_nii.get_fdata().astype(np.float32)

        # Load aneurysm mask if positive
        mask_path = os.path.join(self.loc_masks_dir, f"{scan_id}.nii.gz")
        mask_data = None
        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata().astype(np.float32)
            mask_data = (mask_data > 0).astype(np.float32)

        # Normalize intensity
        norm_vol = self.preprocessor.normalize_intensity(vol_data, modality)

        # Determine center voxel
        if sample["is_aneurysm"] == 1:
            center_voxel = sample["centroid_voxel"]
        else:
            if sample["centroid_voxel"] is None:
                sample["centroid_voxel"] = self._get_negative_centroid(
                    scan_id=scan_id,
                    vol_shape=vol_data.shape,
                    seed=sample.get("sample_seed", idx),
                )
            center_voxel = sample["centroid_voxel"]

        # Extract base patch: with margin if training jitter is used
        base_size = (
            64 + 2 * self.jitter_range,
            64 + 2 * self.jitter_range,
            64 + 2 * self.jitter_range,
        )
        base_vol, base_mask = self.preprocessor.extract_patch(
            norm_vol,
            center_voxel=center_voxel,
            mask=mask_data,
            jitter_range=0,
            patch_size=base_size,
        )

        result = (base_vol, base_mask, center_voxel)
        if self.cache_patches:
            self._patch_cache[idx] = result
        return result

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        base_vol, base_mask, center_voxel = self._load_base_patch(idx)

        # Determine 64^3 patch slice from base patch
        if self.is_training and self.jitter_range > 0:
            jitter = np.random.randint(-self.jitter_range, self.jitter_range + 1, size=3)
            start = self.jitter_range + jitter
            vol_patch = base_vol[start[0] : start[0] + 64, start[1] : start[1] + 64, start[2] : start[2] + 64]
            mask_patch = (
                base_mask[start[0] : start[0] + 64, start[1] : start[1] + 64, start[2] : start[2] + 64]
                if base_mask is not None
                else None
            )
            # Offset points from patch center to true centroid
            if sample["is_aneurysm"] == 1:
                offset_voxel = -jitter.astype(np.float32)
            else:
                offset_voxel = np.zeros(3, dtype=np.float32)

            # 3D Data Augmentation: Random spatial flips along sagittal, coronal, and axial axes
            for axis in (0, 1, 2):
                if np.random.rand() > 0.5:
                    vol_patch = np.flip(vol_patch, axis=axis)
                    if mask_patch is not None:
                        mask_patch = np.flip(mask_patch, axis=axis)
                    if sample["is_aneurysm"] == 1:
                        offset_voxel[axis] = -offset_voxel[axis]

            vol_patch = np.ascontiguousarray(vol_patch)
            if mask_patch is not None:
                mask_patch = np.ascontiguousarray(mask_patch)
        else:
            # Deterministic evaluation: centered slice, no jitter, no flips
            start = self.jitter_range
            vol_patch = base_vol[start : start + 64, start : start + 64, start : start + 64]
            mask_patch = (
                base_mask[start : start + 64, start : start + 64, start : start + 64]
                if base_mask is not None
                else None
            )
            offset_voxel = np.zeros(3, dtype=np.float32)

        # Convert to PyTorch tensors with channel dimension (C, D, H, W)
        patch_tensor = torch.from_numpy(vol_patch).unsqueeze(0).float()
        mask_tensor = (
            torch.from_numpy(mask_patch).unsqueeze(0).float()
            if mask_patch is not None
            else torch.zeros_like(patch_tensor)
        )

        return {
            "patch": patch_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample["is_aneurysm"], dtype=torch.long),
            "size_mm": torch.tensor(sample["max_diameter_mm"], dtype=torch.float32),
            "size_category": sample["size_category"],
            "location_name": sample["location_name"],
            "modality": sample["modality"],
            "center": sample["center"],
            "scan_id": sample["scan_id"],
            "centroid_voxel": torch.tensor(center_voxel, dtype=torch.float32),
            "offset_voxel": torch.tensor(offset_voxel, dtype=torch.float32),
        }
