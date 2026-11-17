"""
3D Non-Maximum Suppression (3D-NMS) for Intracranial Aneurysm Detections.

Clusters and deduplicates overlapping candidate detections in physical 3D space,
preventing multiple redundant false alerts for the same vascular lesion.
Supports physical Euclidean distance clustering, 3D spherical IoU suppression,
and confidence-weighted centroid refinement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


def sphere_volume(radius: float) -> float:
    """Computes volume of a 3D sphere: (4/3) * pi * r^3."""
    return (4.0 / 3.0) * math.pi * (radius ** 3)


def sphere_intersection_volume(r1: float, r2: float, d: float) -> float:
    """
    Computes analytical intersection volume of two 3D spheres of radii r1, r2
    separated by center-to-center distance d.
    """
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        # One sphere is entirely contained inside the other
        return sphere_volume(min(r1, r2))

    # Overlapping spherical cap intersection
    term1 = (r1 + r2 - d) ** 2
    term2 = (d ** 2 + 2 * d * r1 - 3 * (r1 ** 2) + 2 * d * r2 - 3 * (r2 ** 2) + 6 * r1 * r2)
    return (math.pi * term1 * term2) / (12.0 * d)


def spherical_iou(
    c1: Tuple[float, float, float],
    r1: float,
    c2: Tuple[float, float, float],
    r2: float,
) -> float:
    """Computes 3D volumetric Intersection-over-Union (IoU) of two spheres."""
    d = float(np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)))
    v1 = sphere_volume(max(r1, 0.1))
    v2 = sphere_volume(max(r2, 0.1))
    inter = sphere_intersection_volume(max(r1, 0.1), max(r2, 0.1), d)
    union = v1 + v2 - inter
    return inter / max(union, 1e-8)


class NMS3D:
    """
    3D Non-Maximum Suppression processor for clinical aneurysm candidate clusters.
    """

    def __init__(
        self,
        distance_threshold_mm: float = 6.0,
        score_threshold: float = 0.0,
        iou_threshold: Optional[float] = None,
        merge_mode: Literal["highest_score", "weighted_average"] = "highest_score",
    ) -> None:
        """
        Args:
            distance_threshold_mm: Physical distance threshold (mm) for suppressing adjacent hits.
            score_threshold: Minimum confidence score to be considered for suppression.
            iou_threshold: Optional spherical IoU threshold (if provided, suppression uses IoU).
            merge_mode:
                - 'highest_score': keep the exact properties of the peak candidate.
                - 'weighted_average': merge cluster coordinates via confidence-weighted average.
        """
        self.distance_threshold_mm = distance_threshold_mm
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.merge_mode = merge_mode

    def suppress(
        self,
        detections: List[Dict[str, Union[float, Tuple[float, float, float], np.ndarray, str]]],
        score_key: str = "aneurysm_prob",
        coord_key: str = "refined_phys_coord",
        diameter_key: str = "diameter_mm",
    ) -> List[Dict[str, Union[float, Tuple[float, float, float], np.ndarray, str]]]:
        """
        Performs 3D NMS on a list of detection dictionaries.

        Args:
            detections: List of detection candidate dictionaries.
            score_key: Key in dict containing confidence score / probability.
            coord_key: Key containing 3D coordinates (physical mm preferred).
            diameter_key: Key containing estimated diameter in mm.

        Returns:
            Filtered and deduplicated list of detections sorted by confidence.
        """
        if not detections:
            return []

        # 1. Filter by minimum score
        valid = [
            d for d in detections
            if float(d.get(score_key, 0.0)) >= self.score_threshold
        ]
        if not valid:
            return []

        # 2. Sort descending by confidence score
        sorted_dets = sorted(
            valid,
            key=lambda item: float(item.get(score_key, 0.0)),
            reverse=True,
        )

        suppressed = [False] * len(sorted_dets)
        kept_detections: List[Dict] = []

        for i, det_i in enumerate(sorted_dets):
            if suppressed[i]:
                continue

            coord_i = np.array(det_i[coord_key], dtype=np.float32)
            diam_i = float(det_i.get(diameter_key, 4.0))
            score_i = float(det_i.get(score_key, 0.0))
            r_i = max(0.5, diam_i / 2.0)

            cluster_members = [det_i]

            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue

                det_j = sorted_dets[j]
                coord_j = np.array(det_j[coord_key], dtype=np.float32)
                dist = float(np.linalg.norm(coord_i - coord_j))

                is_overlap = False
                if self.iou_threshold is not None:
                    diam_j = float(det_j.get(diameter_key, 4.0))
                    r_j = max(0.5, diam_j / 2.0)
                    iou = spherical_iou(
                        tuple(coord_i), r_i,
                        tuple(coord_j), r_j
                    )
                    is_overlap = (iou >= self.iou_threshold)
                else:
                    # Physical distance suppression (Clinical standard: 6.0 mm)
                    is_overlap = (dist <= self.distance_threshold_mm)

                if is_overlap:
                    suppressed[j] = True
                    cluster_members.append(det_j)

            # Format final kept detection
            if self.merge_mode == "weighted_average" and len(cluster_members) > 1:
                weights = np.array([float(m.get(score_key, 0.0)) for m in cluster_members], dtype=np.float32)
                w_sum = max(np.sum(weights), 1e-6)
                coords = np.array([m[coord_key] for m in cluster_members], dtype=np.float32)
                avg_coord = np.sum(coords * weights[:, np.newaxis], axis=0) / w_sum

                out_det = dict(det_i)
                out_det[coord_key] = tuple(float(c) for c in avg_coord)
                out_det["cluster_size"] = len(cluster_members)
                kept_detections.append(out_det)
            else:
                out_det = dict(det_i)
                out_det["cluster_size"] = len(cluster_members)
                kept_detections.append(out_det)

        return kept_detections


def nms_3d(
    detections: List[Dict],
    distance_threshold_mm: float = 6.0,
    score_threshold: float = 0.0,
    score_key: str = "aneurysm_prob",
    coord_key: str = "refined_phys_coord",
) -> List[Dict]:
    """Convenience functional wrapper for 3D-NMS."""
    nms = NMS3D(distance_threshold_mm=distance_threshold_mm, score_threshold=score_threshold)
    return nms.suppress(detections, score_key=score_key, coord_key=coord_key)

