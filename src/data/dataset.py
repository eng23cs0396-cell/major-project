"""
PyTorch Volumetric Dataset for Intracranial Aneurysm Detection & Patch Training.
Supports balanced sampling between true aneurysm sites (positive) and
normal vessel bifurcation/branching locations (hard negative).
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional, List, Tuple
from src.data.preprocessor import VolumetricPreprocessor


class AneurysmPatchDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        eda_summary_csv: Optional[str] = "artifacts/eda/aneurysm_annotations_summary.csv",
        dataset_root: str = "dataset(topAneu)",
        preprocessor: Optional[VolumetricPreprocessor] = None,
        neg_pos_ratio: float = 2.0, # 2 negative vessel patches per positive patch
        jitter_range: int = 2,
        is_training: bool = True
    ):
        self.dataset_root = dataset_root
        self.preprocessor = preprocessor or VolumetricPreprocessor()
        self.jitter_range = jitter_range if is_training else 0
        self.is_training = is_training
        
        self.images_dir = os.path.join(dataset_root, "images")
        self.loc_masks_dir = os.path.join(dataset_root, "location_masks")
        self.vessel_masks_dir = os.path.join(dataset_root, "vessel_masks")
        self.loc_jsons_dir = os.path.join(dataset_root, "location_jsons")

        # Load split scans
        self.split_df = pd.read_csv(split_csv)
        self.scan_ids = set(self.split_df["scan_id"])

        # Load aneurysm metadata if available
        self.samples = []
        if eda_summary_csv and os.path.exists(eda_summary_csv):
            eda_df = pd.read_csv(eda_summary_csv)
            split_aneurysms = eda_df[eda_df["scan_id"].isin(self.scan_ids)]
            for _, row in split_aneurysms.iterrows():
                self.samples.append({
                    "scan_id": row["scan_id"],
                    "modality": row["modality"],
                    "center": row["center"],
                    "is_aneurysm": 1,
                    "centroid_voxel": (int(row["centroid_voxel_x"]), int(row["centroid_voxel_y"]), int(row["centroid_voxel_z"])),
                    "max_diameter_mm": float(row["max_diameter_mm"]),
                    "size_category": row["size_category"],
                    "location_name": row["location_name"]
                })

        # Also sample negative vessel candidates from healthy and diseased scans
        num_positives = max(1, len(self.samples))
        target_negatives = int(num_positives * neg_pos_ratio)
        scans_for_negs = self.split_df["scan_id"].values
        np.random.seed(42)

        for i in range(target_negatives):
            s_id = np.random.choice(scans_for_negs)
            row = self.split_df[self.split_df["scan_id"] == s_id].iloc[0]
            self.samples.append({
                "scan_id": s_id,
                "modality": row["modality"],
                "center": row["center"],
                "is_aneurysm": 0,
                "centroid_voxel": None, # sampled dynamically from vessel mask
                "max_diameter_mm": 0.0,
                "size_category": "Normal Vessel",
                "location_name": "Normal Branch"
            })

        print(f"Dataset initialized: {len(self.samples)} total candidate patches "
              f"({num_positives} positive, {len(self.samples) - num_positives} negative)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        scan_id = sample["scan_id"]
        modality = sample["modality"]

        # Load full 3D scan
        img_path = os.path.join(self.images_dir, f"{scan_id}_0000.nii.gz")
        img_nii = nib.load(img_path)
        vol_data = img_nii.get_fdata().astype(np.float32)
        spacing = img_nii.header.get_zooms()[:3]

        # Load aneurysm mask if positive
        mask_path = os.path.join(self.loc_masks_dir, f"{scan_id}.nii.gz")
        mask_data = None
        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata().astype(np.float32)
            mask_data = (mask_data > 0).astype(np.float32)

        # Normalize intensity
        norm_vol = self.preprocessor.normalize_intensity(vol_data, modality)

        # Determine patch center. Jitter positives here so the offset head has a defined target.
        if sample["is_aneurysm"] == 1:
            true_centroid = np.array(sample["centroid_voxel"], dtype=np.float32)
            center_voxel = true_centroid.copy()
            if self.jitter_range > 0:
                jitter = np.random.randint(-self.jitter_range, self.jitter_range + 1, size=3)
                center_voxel = center_voxel + jitter
            offset_voxel = true_centroid - center_voxel
            center_voxel = tuple(int(v) for v in np.round(center_voxel))
        else:
            vessel_path = os.path.join(self.vessel_masks_dir, f"{scan_id}.nii.gz")
            if os.path.exists(vessel_path):
                vessel_data = nib.load(vessel_path).get_fdata()
                vessel_pts = np.argwhere(vessel_data > 0)
                if len(vessel_pts) > 0:
                    center_voxel = tuple(int(v) for v in vessel_pts[np.random.choice(len(vessel_pts))])
                else:
                    center_voxel = tuple(int(v) for v in (np.array(vol_data.shape) // 2))
            else:
                center_voxel = tuple(int(v) for v in (np.array(vol_data.shape) // 2))
            offset_voxel = np.zeros(3, dtype=np.float32)

        vol_patch, mask_patch = self.preprocessor.extract_patch(
            norm_vol,
            center_voxel=center_voxel,
            mask=mask_data,
            jitter_range=0,
        )

        # Convert to PyTorch tensors with channel dimension (C, D, H, W)
        patch_tensor = torch.from_numpy(vol_patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).float() if mask_patch is not None else torch.zeros_like(patch_tensor)

        return {
            "patch": patch_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample["is_aneurysm"], dtype=torch.long),
            "size_mm": torch.tensor(sample["max_diameter_mm"], dtype=torch.float32),
            "size_category": sample["size_category"],
            "location_name": sample["location_name"],
            "modality": modality,
            "center": sample["center"],
            "scan_id": scan_id,
            "centroid_voxel": torch.tensor(center_voxel, dtype=torch.float32),
            "offset_voxel": torch.tensor(offset_voxel, dtype=torch.float32),
        }
