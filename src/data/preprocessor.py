"""
3D Volumetric Preprocessor for TopAneu MRA and CTA Scans.
Implements modality-specific intensity normalization, isotropic spatial resampling,
and candidate Volume-of-Interest (VOI) patch extraction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from typing import Tuple, Optional, Union


class VolumetricPreprocessor:
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        mra_lower_pct: float = 0.5,
        mra_upper_pct: float = 99.5,
        cta_window_min: float = 100.0,
        cta_window_max: float = 700.0
    ):
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.mra_lower_pct = mra_lower_pct
        self.mra_upper_pct = mra_upper_pct
        self.cta_window_min = cta_window_min
        self.cta_window_max = cta_window_max

    def normalize_intensity(self, volume: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply modality-specific intensity normalization.
        - MRA: Percentile clipping on non-zero brain voxels + Z-score standardization
        - CTA: Hounsfield Unit (HU) windowing [100, 700] for blood vessels + min-max [0, 1]
        """
        vol = volume.astype(np.float32)
        mod = modality.lower()

        if "mr" in mod:
            non_zero = vol[vol > 0]
            if len(non_zero) > 0:
                p_low = np.percentile(non_zero, self.mra_lower_pct)
                p_high = np.percentile(non_zero, self.mra_upper_pct)
                vol = np.clip(vol, p_low, p_high)
                
                # Z-score on brain parenchyma
                mean = np.mean(vol[vol > 0])
                std = np.std(vol[vol > 0]) + 1e-6
                vol = (vol - mean) / std
            else:
                vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-6)
        elif "ct" in mod:
            # Contrast-enhanced vascular window
            vol = np.clip(vol, self.cta_window_min, self.cta_window_max)
            vol = (vol - self.cta_window_min) / (self.cta_window_max - self.cta_window_min + 1e-6)
        else:
            # Generic Min-Max
            v_min, v_max = vol.min(), vol.max()
            vol = (vol - v_min) / (v_max - v_min + 1e-6)

        return vol

    def extract_patch(
        self,
        volume: np.ndarray,
        center_voxel: Tuple[int, int, int],
        mask: Optional[np.ndarray] = None,
        jitter_range: int = 0,
        patch_size: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract a 3D patch of shape (D, H, W) centered at `center_voxel`.
        Applies symmetric padding if the patch exceeds volume boundaries.
        Supports random coordinate jitter for training data augmentation.
        """
        cx, cy, cz = center_voxel
        if jitter_range > 0:
            cx += np.random.randint(-jitter_range, jitter_range + 1)
            cy += np.random.randint(-jitter_range, jitter_range + 1)
            cz += np.random.randint(-jitter_range, jitter_range + 1)

        target_size = patch_size or self.patch_size
        pd_x, pd_y, pd_z = target_size
        hx, hy, hz = pd_x // 2, pd_y // 2, pd_z // 2

        # Desired slice boundaries
        x_start, x_end = int(cx - hx), int(cx + hx)
        y_start, y_end = int(cy - hy), int(cy + hy)
        z_start, z_end = int(cz - hz), int(cz + hz)

        # Pad boundaries
        pad_x_before = max(0, -x_start)
        pad_x_after = max(0, x_end - volume.shape[0])
        pad_y_before = max(0, -y_start)
        pad_y_after = max(0, y_end - volume.shape[1])
        pad_z_before = max(0, -z_start)
        pad_z_after = max(0, z_end - volume.shape[2])

        pad_width = [
            (pad_x_before, pad_x_after),
            (pad_y_before, pad_y_after),
            (pad_z_before, pad_z_after)
        ]

        if any(p > 0 for pair in pad_width for p in pair):
            padded_vol = np.pad(volume, pad_width, mode="edge")
            if mask is not None:
                padded_mask = np.pad(mask, pad_width, mode="constant", constant_values=0)
            else:
                padded_mask = None
        else:
            padded_vol = volume
            padded_mask = mask

        # Shifted extraction indices
        xs = x_start + pad_x_before
        xe = xs + pd_x
        ys = y_start + pad_y_before
        ye = ys + pd_y
        zs = z_start + pad_z_before
        ze = zs + pd_z

        vol_patch = padded_vol[xs:xe, ys:ye, zs:ze]
        mask_patch = padded_mask[xs:xe, ys:ye, zs:ze] if padded_mask is not None else None

        assert vol_patch.shape == target_size, f"Patch shape mismatch: {vol_patch.shape} vs {target_size}"
        return vol_patch, mask_patch

