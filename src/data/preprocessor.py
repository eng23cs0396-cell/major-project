"""
3D Volumetric Preprocessor for TopAneu MRA and CTA Scans.
Implements modality-specific intensity normalization, isotropic spatial resampling,
and candidate Volume-of-Interest (VOI) patch extraction.
"""

import numpy as np
from typing import Tuple, Optional


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

        - MRA: Percentile clipping on non-zero voxels + Z-score standardization
        - CTA: Hounsfield Unit (HU) windowing [100, 700] + min-max [0, 1]
        """
        vol = volume.astype(np.float32, copy=True)
        mod = modality.lower()

        if "mr" in mod:
            # Create mask once instead of repeatedly doing vol > 0
            mask = vol > 0

            if np.any(mask):
                # Percentiles still require extracting non-zero values.
                # This is only done once.
                non_zero = vol[mask]

                p_low = np.percentile(
                    non_zero,
                    self.mra_lower_pct
                )
                p_high = np.percentile(
                    non_zero,
                    self.mra_upper_pct
                )

                # Release the temporary array before allocating more memory
                del non_zero

                # Clip in-place to reduce memory usage
                np.clip(vol, p_low, p_high, out=vol)

                # Calculate mean/std using the existing mask
                values = vol[mask]

                mean = np.mean(values, dtype=np.float64)
                std = np.std(values, dtype=np.float64) + 1e-6

                del values
                del mask

                # Z-score
                vol = (vol - mean) / std

            else:
                mean = np.mean(vol, dtype=np.float64)
                std = np.std(vol, dtype=np.float64) + 1e-6
                vol = (vol - mean) / std

            return vol

        if "ct" in mod:
            # Contrast-enhanced vascular window
            np.clip(
                vol,
                self.cta_window_min,
                self.cta_window_max,
                out=vol
            )

            vol -= self.cta_window_min
            vol /= (
                self.cta_window_max
                - self.cta_window_min
                + 1e-6
            )

            return vol

        # Generic Min-Max
        v_min = vol.min()
        v_max = vol.max()

        vol -= v_min
        vol /= (v_max - v_min + 1e-6)

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

        # Allocate output patch directly to eliminate full-volume array padding (prevents GB-scale RAM thrashing)
        vol_patch = np.zeros(target_size, dtype=np.float32)
        mask_patch = np.zeros(target_size, dtype=bool) if mask is not None else None

        # Overlapping volume boundaries
        vx_start, vx_end = max(0, x_start), min(volume.shape[0], x_end)
        vy_start, vy_end = max(0, y_start), min(volume.shape[1], y_end)
        vz_start, vz_end = max(0, z_start), min(volume.shape[2], z_end)

        # Overlapping patch boundaries
        px_start, px_end = vx_start - x_start, vx_end - x_start
        py_start, py_end = vy_start - y_start, vy_end - y_start
        pz_start, pz_end = vz_start - z_start, vz_end - z_start

        if vx_start < vx_end and vy_start < vy_end and vz_start < vz_end:
            vol_patch[px_start:px_end, py_start:py_end, pz_start:pz_end] = volume[vx_start:vx_end, vy_start:vy_end, vz_start:vz_end]
            if mask is not None and mask_patch is not None:
                mask_patch[px_start:px_end, py_start:py_end, pz_start:pz_end] = mask[vx_start:vx_end, vy_start:vy_end, vz_start:vz_end]

        if vol_patch.shape != target_size:
            raise ValueError(f"Patch shape mismatch: {vol_patch.shape} vs {target_size}")
        return vol_patch, mask_patch

    def extract_brain_mask(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        modality: str = "mr",
    ) -> np.ndarray:
        """
        Automated Intracranial Skull-Stripping & Brain Cavity Extraction.

        Isolates the intracranial cavity and removes the calvarial skull, scalp subcutaneous
        fat, ocular fat, and background air. Runs in <3 seconds using fast downsampled 3D morphology.

        Args:
            volume: 3D intensity scan (MRA or CTA).
            spacing: Physical voxel spacing in mm.
            modality: 'mr' or 'ct'.

        Returns:
            3D boolean array indicating the intracranial brain cavity.
        """
        from scipy import ndimage
        from skimage.filters import threshold_otsu

        orig_shape = volume.shape
        mod = modality.lower()

        # Downsample to half resolution for rapid 3D morphology
        down_shape = tuple(max(8, s // 2) for s in orig_shape)
        step_x = orig_shape[0] / down_shape[0]
        step_y = orig_shape[1] / down_shape[1]
        step_z = orig_shape[2] / down_shape[2]

        idx_x = np.clip((np.arange(down_shape[0]) * step_x).astype(int), 0, orig_shape[0] - 1)
        idx_y = np.clip((np.arange(down_shape[1]) * step_y).astype(int), 0, orig_shape[1] - 1)
        idx_z = np.clip((np.arange(down_shape[2]) * step_z).astype(int), 0, orig_shape[2] - 1)

        sub_vol = volume[np.ix_(idx_x, idx_y, idx_z)]

        if "ct" in mod:
            # CTA soft tissue / brain parenchyma windowing [10, 90] HU
            brain_sub = (sub_vol >= 10.0) & (sub_vol <= 90.0)
        else:
            # TOF-MRA: Otsu thresholding on positive brain tissue voxels
            pos_voxels = sub_vol[sub_vol > 0]
            if len(pos_voxels) > 100:
                thresh = float(threshold_otsu(pos_voxels)) * 0.4
            else:
                thresh = 10.0
            brain_sub = sub_vol > thresh

        # 3D morphological closing to bridge ventricles and sulci
        struct = ndimage.generate_binary_structure(3, 1)
        closed = ndimage.binary_closing(brain_sub, structure=struct, iterations=3)
        filled = ndimage.binary_fill_holes(closed)

        # Retain largest connected component (intracranial vault)
        labeled, num_features = ndimage.label(filled)
        if num_features > 0:
            sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
            largest_label = int(np.argmax(sizes)) + 1
            brain_cavity = labeled == largest_label
        else:
            brain_cavity = filled

        # Erode 2 voxels (4-5 mm) to strip scalp fat rim and skull marrow
        eroded = ndimage.binary_erosion(brain_cavity, structure=struct, iterations=2)

        # Upsample back to original grid
        upsampled = eroded[np.ix_(
            np.clip((np.arange(orig_shape[0]) / step_x).astype(int), 0, down_shape[0] - 1),
            np.clip((np.arange(orig_shape[1]) / step_y).astype(int), 0, down_shape[1] - 1),
            np.clip((np.arange(orig_shape[2]) / step_z).astype(int), 0, down_shape[2] - 1),
        )]
        return upsampled.astype(bool)

    def extract_automated_vessels(
        self,
        volume: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        modality: str = "mr",
        percentile: float = 99.6,
    ) -> np.ndarray:
        """
        Extracts intracranial cerebral arteries directly from a raw scan volume.
        Eliminates scalp fat, dural sinuses, and calvarial bone artifacts without requiring
        a manual vessel mask file.

        Args:
            volume: 3D intensity array.
            brain_mask: Optional precomputed brain mask. If None, computed on the fly.
            spacing: Physical voxel spacing in mm.
            modality: 'mr' or 'ct'.
            percentile: Percentile threshold for TOF-MRA high-flow vessel signal (default: 99.6).

        Returns:
            3D binary arterial mask.
        """
        from scipy import ndimage

        if brain_mask is None:
            brain_mask = self.extract_brain_mask(volume, spacing=spacing, modality=modality)

        masked_vol = np.where(brain_mask, volume, 0.0)
        intracranial_voxels = masked_vol[brain_mask]

        if len(intracranial_voxels) == 0:
            return np.zeros_like(volume, dtype=bool)

        mod = modality.lower()
        if "ct" in mod:
            # CTA contrast-enhanced vessel window [150, 500] HU within brain parenchyma (excludes hyperdense bone)
            vessel_mask = (masked_vol >= 150.0) & (masked_vol <= 500.0) & brain_mask
        else:
            # TOF-MRA high-velocity flow void signal (top 0.4% isolates true Circle of Willis arteries)
            thresh = float(np.percentile(intracranial_voxels, percentile))
            vessel_mask = (masked_vol >= thresh) & brain_mask

        # Filter out small disconnected noise specks (< 15 voxels)
        labeled, num_features = ndimage.label(vessel_mask)
        if num_features > 0:
            sizes = ndimage.sum(vessel_mask, labeled, range(1, num_features + 1))
            keep_labels = np.where(sizes >= 15)[0] + 1
            vessel_mask = np.isin(labeled, keep_labels)

        return vessel_mask.astype(bool)

