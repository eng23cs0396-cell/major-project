"""
Vascular Topology Module: 3D Medial Axis Skeletonization & Analysis.

Extracts topologically sound 1D centerlines from binary 3D cerebral vessel
segmentations (TOF-MRA and CTA), computes local vessel caliber via Euclidean
Distance Transform (EDT), detects endpoints, waypoints, and bifurcations,
and prunes spurious micro-branch noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


class VesselSkeletonizer:
    """
    Performs 3D morphological skeletonization and topological analysis
    on binary cerebrovascular segmentations.
    """

    def __init__(
        self,
        pruning_threshold: int = 3,
        min_component_size: int = 5,
    ) -> None:
        """
        Args:
            pruning_threshold: Minimum length in voxels for terminal spurs.
            min_component_size: Remove disconnected skeleton clusters smaller than this.
        """
        self.pruning_threshold = pruning_threshold
        self.min_component_size = min_component_size

        # 3D 26-connectivity neighbor kernel (3x3x3 with center 0)
        self._neighbor_kernel = np.ones((3, 3, 3), dtype=np.int32)
        self._neighbor_kernel[1, 1, 1] = 0

    def compute_distance_transform(
        self,
        vessel_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Compute the 3D Euclidean Distance Transform (EDT) representing the
        local physical radius (caliber) in millimeters for every vessel voxel.

        Args:
            vessel_mask: 3D binary array (True/1 for vessel, False/0 for background).
            spacing: Physical voxel spacing in millimeters (dx, dy, dz).

        Returns:
            3D float32 array where each foreground voxel holds its distance
            to the nearest vessel boundary in millimeters.
        """
        mask_bool = vessel_mask > 0
        if not np.any(mask_bool):
            return np.zeros(vessel_mask.shape, dtype=np.float32)

        edt = ndimage.distance_transform_edt(mask_bool, sampling=spacing)
        return edt.astype(np.float32)

    def skeletonize(self, vessel_mask: np.ndarray) -> np.ndarray:
        """
        Extract the 1D medial axis centerline skeleton from a 3D vessel mask.

        Args:
            vessel_mask: 3D binary array (boolean or integer).

        Returns:
            3D boolean array of the same shape where True represents a skeleton voxel.
        """
        mask_bool = vessel_mask > 0
        if not np.any(mask_bool):
            return np.zeros(vessel_mask.shape, dtype=bool)

        # 3D Medial axis thinning
        skeleton = skeletonize(mask_bool)

        # Filter out tiny disconnected noise components
        if self.min_component_size > 1:
            skeleton = self._remove_small_components(skeleton, self.min_component_size)

        # Prune dead-end micro-spurs
        if self.pruning_threshold > 1:
            skeleton = self.prune_spurs(skeleton, max_iterations=self.pruning_threshold)

        return skeleton

    def get_neighbor_counts(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Compute the number of 26-connected neighbors for each skeleton voxel.

        Returns:
            3D int32 array containing neighbor degrees (0 for non-skeleton voxels).
        """
        skel_int = (skeleton > 0).astype(np.int32)
        counts = ndimage.convolve(skel_int, self._neighbor_kernel, mode="constant", cval=0)
        return counts * skel_int

    def classify_nodes(
        self,
        skeleton: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Classify all skeleton voxels into topological categories based on 26-connectivity:
        - Endpoints: exactly 1 neighbor (vessel termination or scan boundary)
        - Waypoints: exactly 2 neighbors (tubular vessel centerline)
        - Junctions / Bifurcations: >= 3 neighbors (arterial branching site)

        Returns:
            Dictionary with keys 'endpoints', 'waypoints', 'junctions', and 'isolated',
            each containing an (N, 3) ndarray of voxel coordinates.
        """
        counts = self.get_neighbor_counts(skeleton)

        isolated = np.argwhere((counts == 0) & (skeleton > 0))
        endpoints = np.argwhere(counts == 1)
        waypoints = np.argwhere(counts == 2)
        junctions = np.argwhere(counts >= 3)

        return {
            "isolated": isolated,
            "endpoints": endpoints,
            "waypoints": waypoints,
            "junctions": junctions,
        }

    def prune_spurs(self, skeleton: np.ndarray, max_iterations: int = 3) -> np.ndarray:
        """
        Iteratively removes short terminal dead-end branches (spurs) that often
        arise from surface irregularities rather than real arterial branches.
        """
        pruned = skeleton.copy()
        for _ in range(max_iterations):
            counts = self.get_neighbor_counts(pruned)
            # Remove endpoints that have only 1 neighbor
            endpoints_mask = (counts == 1)
            if not np.any(endpoints_mask):
                break
            pruned[endpoints_mask] = False

        # Re-check that we haven't eliminated the whole skeleton
        if not np.any(pruned):
            return skeleton
        return pruned

    def _remove_small_components(self, skeleton: np.ndarray, min_size: int) -> np.ndarray:
        """Removes disconnected components smaller than min_size voxels."""
        labeled, num_features = ndimage.label(skeleton, structure=np.ones((3, 3, 3)))
        if num_features <= 1:
            return skeleton

        component_sizes = ndimage.sum_labels(skeleton, labeled, range(1, num_features + 1))
        small_mask = np.isin(labeled, np.where(component_sizes < min_size)[0] + 1)
        cleaned = skeleton.copy()
        cleaned[small_mask] = False
        return cleaned
