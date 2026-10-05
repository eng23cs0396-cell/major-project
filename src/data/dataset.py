"""
PyTorch Volumetric Dataset for Intracranial Aneurysm Detection & Patch Training.
Supports balanced sampling between true aneurysm sites (positive) and
normal vessel bifurcation/branching locations (hard negative).

Includes:
- On-disk patch cache. Full scans are never kept on the dataset.
- Deterministic negative patch evaluation for reproducible validation.
- Anatomic safety margin preventing false-negative supervision on diseased scans.
- 3D spatial flip data augmentation with synchronized offset transformation.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        cache_dir: str = "artifacts/patch_cache",
        min_neg_distance_voxels: float = 20.0,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = preprocessor or VolumetricPreprocessor()
        self.jitter_range = jitter_range if is_training else 0
        self.is_training = is_training
        self.cache_patches = cache_patches
        self.min_neg_distance_voxels = min_neg_distance_voxels
        split_name = os.path.splitext(os.path.basename(split_csv))[0]
        self.cache_dir = os.path.join(cache_dir, f"{split_name}_j{self.jitter_range}")
        self._base_size = (
            64 + 2 * self.jitter_range,
            64 + 2 * self.jitter_range,
            64 + 2 * self.jitter_range,
        )

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

        print(
            f"Dataset initialized ({'Train' if is_training else 'Val/Test'}): "
            f"{len(self.samples)} total candidate patches "
            f"({num_positives} positive, {len(self.samples) - num_positives} negative) | "
            f"Disk cache: {self.cache_patches} ({self.cache_dir})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f"{idx}.npz")

    def _read_disk_cache(self, idx: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]]:
        path = self._cache_path(idx)
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            vol = np.array(data["vol"], dtype=np.float32, copy=True)
            has_mask = bool(data["has_mask"])
            mask = np.array(data["mask"], dtype=np.float32, copy=True) if has_mask else None
            center = tuple(int(v) for v in data["center"])
        if vol.shape != self._base_size:
            return None
        if mask is not None and mask.shape != self._base_size:
            return None
        return vol, mask, center

    def _write_disk_cache(
        self,
        idx: int,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        center: Tuple[int, int, int],
    ) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(idx)
        tmp = path + ".partial.npz"
        has_mask = mask is not None
        np.savez_compressed(
            tmp,
            vol=np.ascontiguousarray(vol, dtype=np.float32),
            mask=np.ascontiguousarray(mask, dtype=np.float32) if has_mask else np.zeros((1,), dtype=np.float32),
            has_mask=np.int8(1 if has_mask else 0),
            center=np.asarray(center, dtype=np.int32),
        )
        os.replace(tmp, path)

    def _read_vessel_points(self, scan_id: str) -> np.ndarray:
        """Return vessel coordinates. The caller must drop this array; it is not stored."""
        vessel_path = os.path.join(self.vessel_masks_dir, f"{scan_id}.nii.gz")
        if not os.path.exists(vessel_path):
            return np.empty((0, 3), dtype=np.int32)
        vessel_nii = nib.load(vessel_path)
        vessel_data = np.asarray(vessel_nii.dataobj, dtype=np.float32)
        del vessel_nii
        pts = np.argwhere(vessel_data > 0)
        del vessel_data
        if scan_id in self.aneurysms_by_scan and len(pts) > 0:
            for ac in self.aneurysms_by_scan[scan_id]:
                dists = np.linalg.norm(pts - np.array(ac), axis=1)
                pts = pts[dists >= self.min_neg_distance_voxels]
        return pts

    def _resolve_center(
        self,
        idx: int,
        vol_shape: Tuple[int, ...],
        vessel_pts: Optional[np.ndarray],
    ) -> Tuple[int, int, int]:
        sample = self.samples[idx]
        if sample["centroid_voxel"] is not None:
            return sample["centroid_voxel"]
        if vessel_pts is not None and len(vessel_pts) > 0:
            local_rng = np.random.RandomState(sample.get("sample_seed", idx))
            choice = local_rng.choice(len(vessel_pts))
            center = tuple(int(v) for v in vessel_pts[choice])
        else:
            center = tuple(int(v) for v in (np.array(vol_shape) // 2))
        sample["centroid_voxel"] = center
        return center

    def _read_normalized(self, scan_id: str, modality: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load one scan into locals. Nothing is attached to the dataset."""
        img_path = os.path.join(self.images_dir, f"{scan_id}_0000.nii.gz")
        img_nii = nib.load(img_path)
        vol_data = np.asarray(img_nii.dataobj, dtype=np.float32)
        del img_nii
        norm_vol = self.preprocessor.normalize_intensity(vol_data, modality)
        del vol_data

        mask_data = None
        mask_path = os.path.join(self.loc_masks_dir, f"{scan_id}.nii.gz")
        if os.path.exists(mask_path):
            mask_nii = nib.load(mask_path)
            mask_data = np.asarray(mask_nii.dataobj, dtype=np.float32)
            del mask_nii
        return norm_vol, mask_data

    def _extract_patch_at(
        self,
        idx: int,
        norm_vol: np.ndarray,
        mask_data: Optional[np.ndarray],
        vessel_pts: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]:
        center_voxel = self._resolve_center(idx, norm_vol.shape, vessel_pts)
        base_vol, base_mask = self.preprocessor.extract_patch(
            norm_vol,
            center_voxel=center_voxel,
            mask=mask_data,
            jitter_range=0,
            patch_size=self._base_size,
        )
        if base_mask is not None:
            base_mask = (base_mask > 0).astype(np.float32, copy=False)
        return base_vol, base_mask, center_voxel

    def _write_scan_patches(self, scan_id: str, indices: List[int]) -> None:
        modality = str(self.samples[indices[0]]["modality"])
        norm_vol, mask_data = self._read_normalized(scan_id, modality)
        needs_vessels = any(
            self.samples[i]["is_aneurysm"] == 0 and self.samples[i]["centroid_voxel"] is None
            for i in indices
        )
        vessel_pts = self._read_vessel_points(scan_id) if needs_vessels else None
        try:
            for idx in indices:
                self._resolve_center(idx, norm_vol.shape, vessel_pts)
            vessel_pts = None
            for idx in indices:
                base_vol, base_mask, center = self._extract_patch_at(idx, norm_vol, mask_data, None)
                self._write_disk_cache(idx, base_vol, base_mask, center)
                del base_vol, base_mask
        finally:
            del norm_vol, mask_data, vessel_pts

    def warm_cache(self, indices: Optional[List[int]] = None) -> None:
        """Write missing patches to disk, one scan at a time, then drop that scan."""
        if not self.cache_patches:
            return
        if indices is None:
            indices = list(range(len(self.samples)))
        missing = [i for i in indices if not os.path.exists(self._cache_path(i))]
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx in missing:
            groups[str(self.samples[idx]["scan_id"])].append(idx)
        for scan_id, idxs in tqdm(groups.items(), desc="Disk cache", leave=False):
            self._write_scan_patches(scan_id, idxs)

    def _load_base_patch(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]:
        if self.cache_patches:
            cached = self._read_disk_cache(idx)
            if cached is not None:
                return cached
            self._write_scan_patches(str(self.samples[idx]["scan_id"]), [idx])
            cached = self._read_disk_cache(idx)
            if cached is None:
                raise RuntimeError(f"Failed to write patch cache for sample {idx}")
            return cached

        sample = self.samples[idx]
        norm_vol, mask_data = self._read_normalized(str(sample["scan_id"]), str(sample["modality"]))
        vessel_pts = None
        try:
            if sample["is_aneurysm"] == 0 and sample["centroid_voxel"] is None:
                vessel_pts = self._read_vessel_points(str(sample["scan_id"]))
            return self._extract_patch_at(idx, norm_vol, mask_data, vessel_pts)
        finally:
            del norm_vol, mask_data, vessel_pts

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
