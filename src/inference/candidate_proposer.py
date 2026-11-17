"""
Clinical Candidate Proposer Module for 3D Cerebral Angiography.

Traverses the patient's 3D vascular skeleton to generate targeted candidate interrogation
points at arterial bifurcations, junctions, and subsampled centerline waypoints.
Replaces computationally prohibitive 3D exhaustive sliding windows across tens of millions
of voxels with ~200-400 clinically high-yield sites per volume.
"""

from __future__ import annotations

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import nibabel as nib
from scipy.spatial import KDTree

from src.topology.skeletonizer import VesselSkeletonizer


class CandidateProposer:
    """
    Identifies high-suspicion candidate 3D sites along the cerebral arterial tree.
    """

    def __init__(
        self,
        skeletonizer: Optional[VesselSkeletonizer] = None,
        waypoint_stride: int = 6,
        include_bifurcations: bool = True,
        include_waypoints: bool = True,
        include_endpoints: bool = False,
        min_candidate_distance_mm: float = 2.0,
        min_edt_radius_mm: float = 0.2,
        adaptive_stride: bool = True,
    ) -> None:
        """
        Args:
            skeletonizer: VesselSkeletonizer instance (or default if None).
            waypoint_stride: Subsampling stride along tubular centerline waypoints.
            include_bifurcations: Whether to include all arterial branching junctions.
            include_waypoints: Whether to include subsampled tubular centerlines.
            include_endpoints: Whether to include vessel terminations.
            min_candidate_distance_mm: Minimum physical distance between candidate centers
                to avoid excessive spatial redundancy.
            min_edt_radius_mm: Minimum vessel caliber required to keep candidate.
            adaptive_stride: If True, uses denser stride (3 voxels) on thin vessels (<=2mm)
                and standard stride (6 voxels) on thick trunks to maximize small aneurysm recall.
        """
        self.skeletonizer = skeletonizer or VesselSkeletonizer(pruning_threshold=3)
        self.waypoint_stride = waypoint_stride
        self.include_bifurcations = include_bifurcations
        self.include_waypoints = include_waypoints
        self.include_endpoints = include_endpoints
        self.min_candidate_distance_mm = min_candidate_distance_mm
        self.min_edt_radius_mm = min_edt_radius_mm
        self.adaptive_stride = adaptive_stride

    def propose_from_mask(
        self,
        vessel_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> List[Dict[str, Union[Tuple[int, int, int], Tuple[float, float, float], str, float]]]:
        """
        Extracts candidate coordinates from a 3D binary vessel mask.

        Args:
            vessel_mask: 3D binary array (True/1 for vessel lumen, False/0 for background).
            spacing: Voxel spacing in millimeters (dx, dy, dz).

        Returns:
            List of candidate dictionaries, each containing:
                - 'voxel_coord': (x, y, z) integer voxel coordinate
                - 'phys_coord': (px, py, pz) physical coordinate in mm
                - 'candidate_type': 'bifurcation', 'waypoint', or 'endpoint'
                - 'radius_mm': local vessel radius in mm
        """
        spacing_arr = np.array(spacing, dtype=np.float32)
        if not np.any(vessel_mask > 0):
            # Fallback center coordinate for empty masks
            center_vox = tuple(s // 2 for s in vessel_mask.shape)
            center_phys = tuple(float(c * s) for c, s in zip(center_vox, spacing))
            return [{
                "voxel_coord": center_vox,
                "phys_coord": center_phys,
                "candidate_type": "fallback",
                "radius_mm": 0.0,
                "bulge_ratio": 1.0,
            }]

        # 1. Compute Euclidean Distance Transform for local caliber
        edt_mm = self.skeletonizer.compute_distance_transform(vessel_mask, spacing=spacing)

        # 2. Compute 1D Medial Axis Skeleton
        skeleton = self.skeletonizer.skeletonize(vessel_mask)
        if not np.any(skeleton):
            center_vox = tuple(s // 2 for s in vessel_mask.shape)
            center_phys = tuple(float(c * s) for c, s in zip(center_vox, spacing))
            return [{
                "voxel_coord": center_vox,
                "phys_coord": center_phys,
                "candidate_type": "fallback",
                "radius_mm": 0.0,
                "bulge_ratio": 1.0,
            }]

        # 3. Classify Skeleton Nodes
        node_groups = self.skeletonizer.classify_nodes(skeleton)
        raw_candidates: List[Tuple[np.ndarray, str]] = []

        if self.include_bifurcations and len(node_groups["junctions"]) > 0:
            for coord in node_groups["junctions"]:
                raw_candidates.append((coord, "bifurcation"))

        if self.include_waypoints and len(node_groups["waypoints"]) > 0:
            waypoints = node_groups["waypoints"]
            if self.adaptive_stride:
                # Caliber-aware adaptive subsampling: sample delicate vessels (<=2mm) more densely
                small_stride = max(2, self.waypoint_stride // 2)
                for idx, coord in enumerate(waypoints):
                    c_tuple = tuple(int(c) for c in coord)
                    r = float(edt_mm[c_tuple])
                    stride_to_use = small_stride if r <= 2.0 else self.waypoint_stride
                    if idx % stride_to_use == 0:
                        raw_candidates.append((coord, "waypoint"))
            else:
                subsampled_waypoints = waypoints[:: self.waypoint_stride]
                for coord in subsampled_waypoints:
                    raw_candidates.append((coord, "waypoint"))

        if self.include_endpoints and len(node_groups["endpoints"]) > 0:
            for coord in node_groups["endpoints"]:
                raw_candidates.append((coord, "endpoint"))

        if not raw_candidates:
            # If all filtered, pick the skeleton voxel with highest EDT
            skel_coords = np.argwhere(skeleton)
            best_idx = np.argmax([edt_mm[tuple(c)] for c in skel_coords])
            raw_candidates.append((skel_coords[best_idx], "centerline"))

        # Build KDTree over full skeleton voxels for fast baseline caliber calculation
        skel_voxels = np.argwhere(skeleton)
        skel_phys_all = skel_voxels * spacing_arr
        skel_tree = KDTree(skel_phys_all) if len(skel_phys_all) > 1 else None

        # 4. Filter by minimum caliber and spatial deduplication
        accepted_candidates: List[Dict[str, Union[Tuple[int, int, int], Tuple[float, float, float], str, float]]] = []
        accepted_phys_coords: List[np.ndarray] = []

        for coord, cand_type in raw_candidates:
            c_tuple = tuple(int(c) for c in coord)
            radius = float(edt_mm[c_tuple])
            if radius < self.min_edt_radius_mm and len(accepted_candidates) > 0:
                continue

            phys = np.array(coord, dtype=np.float32) * spacing_arr

            # Check minimum distance to already accepted candidates (adaptive threshold)
            req_dist = max(1.2, self.min_candidate_distance_mm * 0.75) if (self.adaptive_stride and radius <= 2.0) else self.min_candidate_distance_mm
            if accepted_phys_coords and req_dist > 0.0:
                dist = np.linalg.norm(np.array(accepted_phys_coords) - phys, axis=1)
                if np.min(dist) < req_dist:
                    continue

            # Compute Murray's focal saccular outpouching ratio relative to supporting parent vessel
            # Evaluates the surrounding vascular skeleton in a 2.5 mm to 7.0 mm ring
            bulge_ratio = 1.0
            if skel_tree is not None:
                nbrs_wide = skel_tree.query_ball_point(phys, r=7.0)
                nbrs_close = skel_tree.query_ball_point(phys, r=2.5)
                ring_indices = [idx for idx in nbrs_wide if idx not in nbrs_close]
                if len(ring_indices) >= 3:
                    ring_radii = [float(edt_mm[tuple(skel_voxels[i])]) for i in ring_indices]
                    # Parent vessel caliber: 90th percentile of surrounding arterial network
                    parent_radius = float(np.percentile(ring_radii, 90))
                    bulge_ratio = float(radius / max(parent_radius, 0.40))
                elif len(nbrs_wide) >= 3:
                    nbr_radii = [float(edt_mm[tuple(skel_voxels[i])]) for i in nbrs_wide]
                    baseline_radius = float(np.percentile(nbr_radii, 90))
                    bulge_ratio = float(radius / max(baseline_radius, 0.40))

            accepted_phys_coords.append(phys)
            accepted_candidates.append({
                "voxel_coord": (int(coord[0]), int(coord[1]), int(coord[2])),
                "phys_coord": (float(phys[0]), float(phys[1]), float(phys[2])),
                "candidate_type": cand_type,
                "radius_mm": radius,
                "bulge_ratio": round(bulge_ratio, 3),
            })

        return accepted_candidates

    def propose_from_file(
        self,
        vessel_mask_path: str,
    ) -> List[Dict[str, Union[Tuple[int, int, int], Tuple[float, float, float], str, float]]]:
        """Loads a NIfTI vessel mask file and generates candidate proposals."""
        nii = nib.load(vessel_mask_path)
        mask = np.asarray(nii.dataobj, dtype=np.uint8) > 0
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
        return self.propose_from_mask(mask, spacing=spacing)

    def evaluate_candidate_recall(
        self,
        candidates: List[Dict],
        ground_truth_centroids: List[Tuple[float, float, float]],
        hit_threshold_mm: float = 6.0,
    ) -> Dict[str, float]:
        """
        Measures the spatial coverage/recall of the candidate generation step
        against known ground-truth aneurysm physical coordinates.

        Args:
            candidates: List of candidate dicts with 'phys_coord'.
            ground_truth_centroids: List of ground-truth physical (x, y, z) in mm.
            hit_threshold_mm: Maximum allowable physical distance to count as a hit.

        Returns:
            Dictionary with candidate recall, hit count, and total ground truth count.
        """
        if not ground_truth_centroids:
            return {"candidate_recall": 1.0, "hits": 0, "total_gt": 0}

        if not candidates:
            return {"candidate_recall": 0.0, "hits": 0, "total_gt": len(ground_truth_centroids)}

        cand_phys = np.array([c["phys_coord"] for c in candidates], dtype=np.float32)
        tree = KDTree(cand_phys)

        hits = 0
        for gt in ground_truth_centroids:
            dist, _ = tree.query(np.array(gt, dtype=np.float32))
            if dist <= hit_threshold_mm:
                hits += 1

        recall = hits / len(ground_truth_centroids)
        return {
            "candidate_recall": recall,
            "hits": hits,
            "total_gt": len(ground_truth_centroids),
        }
