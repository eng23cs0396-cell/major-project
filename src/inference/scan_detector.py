"""
Full-Scan 3D Intracranial Aneurysm Detector Engine.

Coordinates the end-to-end clinical inference pipeline:
1. 3D Medial Axis candidate interrogation points via CandidateProposer.
2. Modality-specific intensity normalization (MRA / CTA).
3. Batched volumetric patch extraction & local k-hop arterial graph extraction.
4. Deep neural model evaluation (3D ResNet Baseline or Multimodal Dual-Branch Fusion).
5. 3D spatial Non-Maximum Suppression (3D-NMS) to eliminate duplicate candidate alerts.
"""

from __future__ import annotations

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import nibabel as nib
import torch
from torch_geometric.data import Batch, Data

from src.inference.candidate_proposer import CandidateProposer
from src.inference.nms3d import NMS3D
from src.data.preprocessor import VolumetricPreprocessor
from src.topology.graph_builder import ArterialGraphBuilder
from src.topology.subgraph_extractor import VesselSubgraphExtractor
from src.models.resnet3d import ResNet3DBaseline
from src.models.dual_branch_detector import DualBranchDetector


class ScanDetector:
    """
    End-to-end full-volume aneurysm detector for uncropped clinical scans.
    """

    def __init__(
        self,
        model_type: Literal["baseline", "fusion"] = "fusion",
        checkpoint_path: Optional[str] = None,
        mra_checkpoint: Optional[str] = "checkpoints/best_fusion_mra.pt",
        cta_checkpoint: Optional[str] = "checkpoints/best_fusion_cta.pt",
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 16,
        nms_distance_mm: float = 6.0,
        conf_threshold: float = 0.05,
        graph_cache_dir: str = "artifacts/graph_cache",
    ) -> None:
        """
        Args:
            model_type: 'baseline' (ResNet3D) or 'fusion' (DualBranchDetector).
            checkpoint_path: Explicit path to .pt model weights. If None, enables
                automatic Dual-Expert routing based on scan modality.
            mra_checkpoint: Specialized model checkpoint for TOF-MRA scans.
            cta_checkpoint: Specialized model checkpoint for CTA scans.
            device: 'cuda' or 'cpu'. Defaults to CUDA if available.
            batch_size: Inference mini-batch size.
            nms_distance_mm: 3D-NMS clustering radius in mm.
            conf_threshold: Candidate score cutoff before NMS.
            graph_cache_dir: Directory holding precomputed scan graphs.
        """
        self.model_type = model_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.nms_distance_mm = nms_distance_mm
        self.conf_threshold = conf_threshold
        self.graph_cache_dir = graph_cache_dir
        self.checkpoint_path = checkpoint_path
        self.mra_checkpoint = mra_checkpoint
        self.cta_checkpoint = cta_checkpoint
        self._loaded_models: Dict[str, torch.nn.Module] = {}
        os.makedirs(self.graph_cache_dir, exist_ok=True)

        self.preprocessor = VolumetricPreprocessor()
        self.proposer = CandidateProposer(waypoint_stride=6, min_candidate_distance_mm=2.0)
        self.nms = NMS3D(distance_threshold_mm=nms_distance_mm, score_threshold=conf_threshold)

        self.graph_builder = ArterialGraphBuilder()
        self.subgraph_extractor = VesselSubgraphExtractor(k_hops=3, max_nodes=64)

        # Initialize model architecture
        if self.model_type == "baseline":
            self.model = ResNet3DBaseline(backbone="resnet18_3d", in_channels=1)
            default_ckpt = "checkpoints/best_baseline_resnet3d.pt"
        else:
            self.model = DualBranchDetector(backbone="resnet18_3d", in_channels=1)
            default_ckpt = "checkpoints/best_fusion_model.pt"

        ckpt_to_load = checkpoint_path or default_ckpt
        if os.path.exists(ckpt_to_load):
            ckpt = torch.load(ckpt_to_load, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print(f"Loaded {self.model_type} model weights from {ckpt_to_load}")
        else:
            print(f"Warning: Checkpoint {ckpt_to_load} not found. Running with uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def _get_active_model(self, modality: str) -> torch.nn.Module:
        """Routes scan to the corresponding modality-specialized Dual-Expert model."""
        if self.model_type == "baseline" or self.checkpoint_path is not None:
            return self.model

        mod = str(modality).lower()
        if mod in ["ct", "cta"] and self.cta_checkpoint and os.path.exists(self.cta_checkpoint):
            target_ckpt = self.cta_checkpoint
        elif mod in ["mr", "mra"] and self.mra_checkpoint and os.path.exists(self.mra_checkpoint):
            target_ckpt = self.mra_checkpoint
        else:
            return self.model

        if target_ckpt not in self._loaded_models:
            expert = DualBranchDetector(backbone="resnet18_3d", in_channels=1)
            ckpt = torch.load(target_ckpt, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            expert.load_state_dict(state_dict)
            expert.to(self.device).eval()
            self._loaded_models[target_ckpt] = expert
            print(f"Loaded {mod.upper()} Expert model from {target_ckpt}")

        return self._loaded_models[target_ckpt]

    def _get_or_build_graph(self, scan_id: str, vessel_mask: np.ndarray, spacing: Tuple[float, float, float]) -> Data:
        """Loads cached graph or constructs it on the fly."""
        cache_path = os.path.join(self.graph_cache_dir, f"{scan_id}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        graph = self.graph_builder.build_graph_from_mask(vessel_mask)
        try:
            torch.save(graph, cache_path)
        except Exception:
            pass
        return graph

    @torch.no_grad()
    def detect_scan(
        self,
        image_volume: np.ndarray,
        vessel_mask: np.ndarray,
        modality: str = "mr",
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        scan_id: str = "scan_001",
    ) -> List[Dict]:
        """
        Runs the full detection pipeline on a single 3D volume.

        Args:
            image_volume: 3D intensity array (raw MRA or CTA).
            vessel_mask: 3D binary vessel mask.
            modality: 'mr' or 'ct'.
            spacing: Voxel spacing in mm (dx, dy, dz).
            scan_id: Identifier string for caching.

        Returns:
            List of detected aneurysms sorted by confidence after 3D-NMS.
        """
        spacing_arr = np.array(spacing, dtype=np.float32)

        # 1. Intensity Normalization
        norm_vol = self.preprocessor.normalize_intensity(image_volume, modality=modality)

        # 2. Topology Candidate Proposals
        candidates = self.proposer.propose_from_mask(vessel_mask, spacing=spacing)
        if not candidates:
            return []

        # 3. For Multimodal Model, acquire full arterial graph and KDTree
        scan_graph = None
        scan_tree = None
        if self.model_type == "fusion":
            scan_graph = self._get_or_build_graph(scan_id, vessel_mask, spacing)
            if scan_graph.pos is not None and scan_graph.num_nodes > 1:
                from scipy.spatial import KDTree
                scan_tree = KDTree(scan_graph.pos.cpu().numpy())

        # 4. Batch Patch & Subgraph Extraction
        raw_detections: List[Dict] = []
        num_candidates = len(candidates)

        for start_idx in range(0, num_candidates, self.batch_size):
            batch_cands = candidates[start_idx : start_idx + self.batch_size]

            patch_list: List[torch.Tensor] = []
            subgraph_list: List[Data] = []
            valid_batch_cands: List[Dict] = []

            for cand in batch_cands:
                vx, vy, vz = cand["voxel_coord"]
                patch, _ = self.preprocessor.extract_patch(
                    norm_vol,
                    center_voxel=(vx, vy, vz),
                    patch_size=(64, 64, 64),
                )
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()  # (1, 64, 64, 64)
                patch_list.append(patch_tensor)
                valid_batch_cands.append(cand)

                if self.model_type == "fusion" and scan_graph is not None:
                    # Extract k-hop local subgraph around candidate matching graph coordinate space (0.5mm isotropic)
                    cand_graph_pos = np.array(cand["voxel_coord"], dtype=np.float32) * 0.5
                    subg = self.subgraph_extractor.extract_subgraph_around_point(
                        scan_graph, cand_graph_pos, tree=scan_tree
                    )
                    subgraph_list.append(subg)

            if not patch_list:
                continue

            batch_patches = torch.stack(patch_list, dim=0).to(self.device)  # (B, 1, 64, 64, 64)

            # Model Forward Pass (Dual-Expert Routed)
            active_model = self._get_active_model(modality)
            if self.model_type == "baseline":
                out = active_model(batch_patches)
            else:
                batch_pyg = Batch.from_data_list(subgraph_list).to(self.device)
                out = active_model(batch_patches, batch_pyg)

            probs = out["aneurysm_prob"].detach().cpu().numpy()
            offsets = out["offset"].detach().cpu().numpy()
            diams = out["diameter_mm"].detach().cpu().numpy()

            if probs.ndim == 0:
                probs = np.array([probs])
                offsets = np.expand_dims(offsets, axis=0)
                diams = np.array([diams])

            for k, cand in enumerate(valid_batch_cands):
                prob_k = float(probs[k])
                offset_k = offsets[k]
                diam_k = float(diams[k])

                # Trust neural network and topological GAT representations
                # Downweight flat non-dilated centerlines only when network confidence is low
                bulge = float(cand.get("bulge_ratio", 1.0))
                if cand.get("candidate_type") == "waypoint" and bulge < 1.02 and prob_k < 0.50:
                    prob_k = prob_k * 0.90

                vx, vy, vz = cand["voxel_coord"]
                ref_vx = float(vx + offset_k[0])
                ref_vy = float(vy + offset_k[1])
                ref_vz = float(vz + offset_k[2])

                ref_px = float(ref_vx * spacing_arr[0])
                ref_py = float(ref_vy * spacing_arr[1])
                ref_pz = float(ref_vz * spacing_arr[2])

                raw_detections.append({
                    "scan_id": scan_id,
                    "voxel_coord": (ref_vx, ref_vy, ref_vz),
                    "phys_coord": (ref_px, ref_py, ref_pz),
                    "candidate_voxel_coord": (vx, vy, vz),
                    "candidate_phys_coord": cand["phys_coord"],
                    "candidate_type": cand["candidate_type"],
                    "bulge_ratio": bulge,
                    "refined_voxel_coord": (ref_vx, ref_vy, ref_vz),
                    "refined_phys_coord": (ref_px, ref_py, ref_pz),
                    "offset_voxels": (float(offset_k[0]), float(offset_k[1]), float(offset_k[2])),
                    "aneurysm_prob": prob_k,
                    "diameter_mm": diam_k,
                })

        # 5. Apply 3D Non-Maximum Suppression
        final_detections = self.nms.suppress(
            raw_detections,
            score_key="aneurysm_prob",
            coord_key="refined_phys_coord",
            diameter_key="diameter_mm",
        )

        return final_detections
