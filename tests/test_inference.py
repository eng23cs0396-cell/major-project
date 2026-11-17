"""
Unit tests for Phase 6: Clinical Inference, 3D-NMS & FROC Evaluation.
"""

from __future__ import annotations

import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.inference.candidate_proposer import CandidateProposer
from src.inference.nms3d import NMS3D, sphere_volume, sphere_intersection_volume, spherical_iou, nms_3d
from src.inference.froc_evaluator import FROCEvaluator
from src.inference.scan_detector import ScanDetector


class TestCandidateProposer:
    def test_synthetic_mask_candidates(self):
        # Create a synthetic 3D cross (bifurcation junction at 16, 16, 16)
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[16, :, 16] = 1  # Line along Y
        mask[16, 16, :] = 1  # Line along Z
        mask[:, 16, 16] = 1  # Line along X

        proposer = CandidateProposer(waypoint_stride=4, min_candidate_distance_mm=1.0)
        cands = proposer.propose_from_mask(mask, spacing=(0.5, 0.5, 0.5))

        assert len(cands) > 0, "Expected at least one candidate proposal"
        # Check keys
        first = cands[0]
        assert "voxel_coord" in first
        assert "phys_coord" in first
        assert "candidate_type" in first
        assert "radius_mm" in first

        # Check candidate recall against the true junction
        gt_junction = [(16 * 0.5, 16 * 0.5, 16 * 0.5)]
        recall_res = proposer.evaluate_candidate_recall(cands, gt_junction, hit_threshold_mm=3.0)
        assert recall_res["candidate_recall"] == 1.0
        assert recall_res["hits"] == 1

    def test_empty_mask_fallback(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        proposer = CandidateProposer()
        cands = proposer.propose_from_mask(mask)
        assert len(cands) == 1
        assert cands[0]["candidate_type"] == "fallback"
        assert cands[0]["voxel_coord"] == (10, 10, 10)


class TestNMS3D:
    def test_sphere_geometry(self):
        v = sphere_volume(3.0)
        expected = (4.0 / 3.0) * np.pi * 27.0
        assert pytest.approx(v, 0.01) == expected

        # Identical spheres: IoU should be 1.0
        iou = spherical_iou((0, 0, 0), 2.0, (0, 0, 0), 2.0)
        assert pytest.approx(iou, 0.01) == 1.0

        # Disjoint spheres: IoU should be 0.0
        iou_disjoint = spherical_iou((0, 0, 0), 2.0, (10, 0, 0), 2.0)
        assert iou_disjoint == 0.0

    def test_nms_suppression(self):
        # 3 detections: two very close (< 2mm apart), one far away (25mm)
        dets = [
            {
                "refined_phys_coord": (10.0, 10.0, 10.0),
                "aneurysm_prob": 0.95,
                "diameter_mm": 4.5,
            },
            {
                "refined_phys_coord": (11.0, 10.5, 10.0),  # Distance ~1.1mm -> should be suppressed
                "aneurysm_prob": 0.80,
                "diameter_mm": 4.0,
            },
            {
                "refined_phys_coord": (35.0, 35.0, 35.0),  # Far away -> should be kept
                "aneurysm_prob": 0.88,
                "diameter_mm": 5.0,
            },
        ]

        nms = NMS3D(distance_threshold_mm=6.0)
        kept = nms.suppress(dets)

        assert len(kept) == 2, f"Expected 2 detections after NMS, got {len(kept)}"
        assert kept[0]["aneurysm_prob"] == 0.95
        assert kept[0]["cluster_size"] == 2
        assert kept[1]["aneurysm_prob"] == 0.88
        assert kept[1]["cluster_size"] == 1

    def test_weighted_average_merge(self):
        dets = [
            {"refined_phys_coord": (10.0, 10.0, 10.0), "aneurysm_prob": 0.8},
            {"refined_phys_coord": (12.0, 10.0, 10.0), "aneurysm_prob": 0.8},
        ]
        nms = NMS3D(distance_threshold_mm=5.0, merge_mode="weighted_average")
        kept = nms.suppress(dets)
        assert len(kept) == 1
        # Midpoint should be (11.0, 10.0, 10.0)
        assert pytest.approx(kept[0]["refined_phys_coord"][0], 0.01) == 11.0


class TestFROCEvaluator:
    def test_froc_synthetic(self):
        # Ground truth: 2 aneurysms
        gt_df = pd.DataFrame([
            {
                "scan_id": "scan_1",
                "centroid_voxel_x": 20,
                "centroid_voxel_y": 20,
                "centroid_voxel_z": 20,
                "spacing_x": 0.5,
                "spacing_y": 0.5,
                "spacing_z": 0.5,
                "max_diameter_mm": 2.5,  # Very Small (<3mm)
            },
            {
                "scan_id": "scan_1",
                "centroid_voxel_x": 60,
                "centroid_voxel_y": 60,
                "centroid_voxel_z": 60,
                "spacing_x": 0.5,
                "spacing_y": 0.5,
                "spacing_z": 0.5,
                "max_diameter_mm": 4.5,  # Small (3-5mm)
            },
        ])

        # Detections:
        # Hit 1: matches GT 1 at (10.0, 10.0, 10.0) mm
        # Hit 2: matches GT 2 at (30.0, 30.0, 30.0) mm
        # FP 1: false alert at (80.0, 80.0, 80.0) mm
        dets = [
            {
                "scan_id": "scan_1",
                "refined_phys_coord": (10.5, 10.0, 10.0),  # dist = 0.5mm -> HIT
                "aneurysm_prob": 0.90,
            },
            {
                "scan_id": "scan_1",
                "refined_phys_coord": (30.0, 31.0, 30.0),  # dist = 1.0mm -> HIT
                "aneurysm_prob": 0.70,
            },
            {
                "scan_id": "scan_1",
                "refined_phys_coord": (80.0, 80.0, 80.0),  # dist > 50mm -> FP
                "aneurysm_prob": 0.50,
            },
        ]

        evaluator = FROCEvaluator(hit_distance_mm=5.0, benchmark_fps=[0.5, 1.0, 2.0])
        res = evaluator.evaluate_detections(dets, gt_df, total_scans=1)

        assert res["total_ground_truth"] == 2
        assert res["gt_counts"]["very_small_under_3mm"] == 1
        assert res["gt_counts"]["small_3_to_5mm"] == 1

        # At threshold >= 0.60: 2 hits, 0 FPs -> Sens = 1.0, FP/scan = 0.0
        # At threshold >= 0.40: 2 hits, 1 FP -> Sens = 1.0, FP/scan = 1.0
        assert res["benchmark_sensitivities"][1.0] == 1.0


class TestScanDetectorEndToEnd:
    def test_mock_scan_detection(self):
        # Create a small synthetic scan and vessel mask
        vol = np.random.randn(40, 40, 40).astype(np.float32)
        mask = np.zeros((40, 40, 40), dtype=np.uint8)
        mask[20, :, 20] = 1
        mask[:, 20, 20] = 1

        detector = ScanDetector(
            model_type="baseline",
            device="cpu",
            batch_size=4,
            conf_threshold=0.0,
        )

        dets = detector.detect_scan(
            image_volume=vol,
            vessel_mask=mask,
            modality="mr",
            spacing=(1.0, 1.0, 1.0),
            scan_id="test_synthetic_scan",
        )

        assert isinstance(dets, list)
        if len(dets) > 0:
            assert "refined_phys_coord" in dets[0]
            assert "aneurysm_prob" in dets[0]
            assert "diameter_mm" in dets[0]

