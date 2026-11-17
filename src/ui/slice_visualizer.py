"""
Orthogonal Slice Visualizer & Multiplanar Reconstruction (MPR) for Neuroimaging.

Generates synchronized Axial, Coronal, Sagittal cross-sectional views and Maximum
Intensity Projections (MIP) centered on detected intracranial aneurysms.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SliceVisualizer:
    """
    Renders annotated orthogonal 2D cross-sections (Axial, Coronal, Sagittal)
    and Maximum Intensity Projections for clinical decision support.
    """

    @staticmethod
    def window_intensity(
        image_slice: np.ndarray,
        modality: str = "mr",
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies radiological Window/Level (W/L) transfer function.
        """
        mod = str(modality).lower()
        arr = np.asarray(image_slice, dtype=np.float32)

        if mod in ["ct", "cta"]:
            # Standard cerebral CTA angio window (W: 600, L: 200)
            wc = window_center if window_center is not None else 200.0
            ww = window_width if window_width is not None else 600.0
            lower = wc - ww / 2.0
            upper = wc + ww / 2.0
        else:
            # TOF-MRA percentile windowing
            p_low = np.percentile(arr, 1.0)
            p_high = np.percentile(arr, 99.2)
            if p_high <= p_low:
                p_high = p_low + 1.0
            lower = p_low
            upper = p_high

        clipped = np.clip(arr, lower, upper)
        norm = (clipped - lower) / (upper - lower + 1e-6)
        return (norm * 255.0).astype(np.uint8)

    @classmethod
    def render_tripartite_view(
        cls,
        volume: np.ndarray,
        center_voxel: Tuple[int, int, int],
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        diameter_mm: Optional[float] = None,
        score: Optional[float] = None,
        location: Optional[str] = None,
        modality: str = "mr",
        crop_radius_voxels: int = 40,
    ) -> Image.Image:
        """
        Renders a composite tripartite image (Axial, Coronal, Sagittal) centered
        on the detected aneurysm centroid with radiological annotations.

        Args:
            volume: 3D numpy array of the brain scan.
            center_voxel: (vx, vy, vz) coordinate of lesion.
            spacing: (dx, dy, dz) voxel spacing in mm.
            diameter_mm: Estimated diameter in mm.
            score: Confidence probability.
            location: Anatomical label.
            modality: 'mr' or 'ct'.
            crop_radius_voxels: Radius of the cropped sub-window around the lesion.

        Returns:
            PIL RGB Image containing the 3 orthogonal views with bounding circles.
        """
        vx, vy, vz = [int(round(c)) for c in center_voxel]
        shape = volume.shape

        # Clamping to valid range
        vx = max(0, min(shape[0] - 1, vx))
        vy = max(0, min(shape[1] - 1, vy))
        vz = max(0, min(shape[2] - 1, vz))

        # 1. Extract 2D orthogonal slices
        # Axial: Slice along Z -> (X, Y) plane
        axial_slice = volume[:, :, vz]
        # Coronal: Slice along Y -> (X, Z) plane
        coronal_slice = volume[:, vy, :]
        # Sagittal: Slice along X -> (Y, Z) plane
        sagittal_slice = volume[vx, :, :]

        # 2. Window and Level
        axial_u8 = cls.window_intensity(axial_slice, modality=modality)
        coronal_u8 = cls.window_intensity(coronal_slice, modality=modality)
        sagittal_u8 = cls.window_intensity(sagittal_slice, modality=modality)

        # 3. Create High-Resolution Matplotlib Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="#0f172a")

        views = [
            ("Axial (XY Plane)", axial_u8, (vx, vy), (spacing[0], spacing[1]), "Z"),
            ("Coronal (XZ Plane)", coronal_u8, (vx, vz), (spacing[0], spacing[2]), "Y"),
            ("Sagittal (YZ Plane)", sagittal_u8, (vy, vz), (spacing[1], spacing[2]), "X"),
        ]

        # Calculate bounding radius in mm
        r_mm = (diameter_mm / 2.0) if (diameter_mm and diameter_mm > 0) else 3.0
        # Give visual clearance around lesion
        draw_r_mm = max(2.5, r_mm * 1.35)

        for ax, (title, img_2d, (cx, cy), (spx, spy), slice_axis) in zip(axes, views):
            ax.set_facecolor("#0b0f19")

            # Determine crop box centered at (cx, cy)
            x_min = max(0, cx - crop_radius_voxels)
            x_max = min(img_2d.shape[0], cx + crop_radius_voxels)
            y_min = max(0, cy - crop_radius_voxels)
            y_max = min(img_2d.shape[1], cy + crop_radius_voxels)

            crop_img = img_2d[x_min:x_max, y_min:y_max]
            # Transpose for radiological orientation (Y vertical, X horizontal)
            display_crop = np.rot90(crop_img)

            # Display with aspect ratio correction
            aspect = spy / max(spx, 1e-4)
            ax.imshow(display_crop, cmap="bone", aspect=aspect)

            # Plot lesion target crosshairs and marker
            # Center in cropped coordinates
            rel_cx = cx - x_min
            rel_cy = cy - y_min
            # Rotated 90 deg counter-clockwise coordinates
            rot_x = rel_cy
            rot_y = crop_img.shape[0] - 1 - rel_cx

            # Radius in pixels for display
            r_px = draw_r_mm / max(spx, 0.1)
            circ = patches.Circle(
                (rot_x, rot_y),
                radius=r_px,
                fill=False,
                edgecolor="#f43f5e",
                linewidth=2.2,
                linestyle="--",
            )
            ax.add_patch(circ)

            # Centroid crosshair
            ax.plot([rot_x - 5, rot_x + 5], [rot_y, rot_y], color="#38bdf8", lw=1.5)
            ax.plot([rot_x, rot_x], [rot_y - 5, rot_y + 5], color="#38bdf8", lw=1.5)

            # Labels and Titles
            ax.set_title(
                f"{title} | Slice: {int(round(center_voxel[{'Z':2, 'Y':1, 'X':0}[slice_axis]]))}",
                color="#f8fafc",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )
            ax.axis("off")

        # Global Title Annotation
        score_str = f"P(Aneurysm): {score*100:.1f}%" if score is not None else ""
        diam_str = f"Diam: {diameter_mm:.1f} mm" if diameter_mm is not None else ""
        loc_str = f"Loc: {location}" if location else ""
        meta_str = " | ".join(filter(None, [diam_str, score_str, loc_str]))

        fig.suptitle(
            f"Multiplanar Aneurysm Target Visualization  [{meta_str}]",
            color="#38bdf8",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    @classmethod
    def render_overview_mip(
        cls,
        volume: np.ndarray,
        detections: list[dict],
        modality: str = "mr",
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> Image.Image:
        """
        Renders an Axial Maximum Intensity Projection (MIP) overview of the entire brain,
        marking all detected aneurysm candidates with labeled markers.
        """
        # Axial MIP along Z axis
        mip_arr = np.max(volume, axis=2)
        mip_u8 = cls.window_intensity(mip_arr, modality=modality)
        mip_rot = np.rot90(mip_u8)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0f172a")
        aspect = spacing[1] / max(spacing[0], 1e-4)
        ax.imshow(mip_rot, cmap="inferno" if modality.lower() in ["ct", "cta"] else "magma", aspect=aspect)

        shape_x, shape_y = mip_arr.shape
        # Sort detections by probability descending
        sorted_dets = sorted(detections, key=lambda d: float(d.get("aneurysm_prob", 0.0)), reverse=True)
        max_labels_to_draw = 6

        for i, det in enumerate(sorted_dets):
            coord = det.get("refined_voxel_coord") or det.get("voxel_coord")
            if coord is None:
                phys = det.get("refined_phys_coord", (0.0, 0.0, 0.0))
                coord = (phys[0] / max(spacing[0], 1e-4), phys[1] / max(spacing[1], 1e-4), phys[2] / max(spacing[2], 1e-4))
            vx, vy = int(round(coord[0])), int(round(coord[1]))
            rot_x = vy
            rot_y = shape_x - 1 - vx
            prob = float(det.get("aneurysm_prob", 0.0))

            # Only label top high-confidence candidates to avoid visual clutter
            if i < max_labels_to_draw and prob >= 0.40:
                circ = patches.Circle(
                    (rot_x, rot_y),
                    radius=7.0,
                    fill=False,
                    edgecolor="#22c55e" if prob > 0.70 else "#eab308",
                    linewidth=2.5,
                )
                ax.add_patch(circ)
                ax.text(
                    rot_x + 9,
                    rot_y - 2,
                    f"#{i+1} ({prob*100:.0f}%)",
                    color="#ffffff",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(facecolor="#1e293b", alpha=0.85, edgecolor="none", pad=2),
                )
            elif prob >= 0.50:
                circ = patches.Circle((rot_x, rot_y), radius=3.5, fill=True, facecolor="#38bdf8", alpha=0.7)
                ax.add_patch(circ)

        if not detections:
            title_text = "Whole-Brain Maximum Intensity Projection (MIP) - Normal Vasculature (Negative Scan)"
        else:
            title_text = f"Whole-Brain Maximum Intensity Projection (MIP) - {len(detections)} Finding(s)"

        ax.set_title(
            title_text,
            color="#f8fafc",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
