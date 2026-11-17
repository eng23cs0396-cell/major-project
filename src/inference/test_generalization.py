"""
Zero-Shot Cross-Center Generalization Evaluation (Center 4 Japan CTA Cohort).

Evaluates models trained on MRA/CTA (Centers 1-3) on held-out Center 4 CTA scans
to quantify cross-center and cross-modality domain shift resilience.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from tqdm import tqdm

from src.inference.scan_detector import ScanDetector
from src.inference.froc_evaluator import FROCEvaluator


def run_cross_center_generalization(
    split_csv: str = "configs/center4_holdout_split.csv",
    eda_summary_csv: str = "artifacts/eda/aneurysm_annotations_summary.csv",
    dataset_root: str = "dataset(topAneu)",
    baseline_ckpt: str = "checkpoints/best_baseline_resnet3d.pt",
    fusion_ckpt: Optional[str] = None,
    output_dir: str = "artifacts/generalization",
    max_scans: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Runs zero-shot generalization benchmark on Center 4 held-out CTA scans.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Center 4 held-out split
    split_df = pd.read_csv(split_csv)
    test_c4 = split_df[split_df["split"] == "test_holdout_center4"].copy()
    if max_scans is not None:
        test_c4 = test_c4.head(max_scans)

    print(f"Loaded {len(test_c4)} Center-4 CTA scans for zero-shot evaluation.")

    # 2. Load ground-truth annotations
    eda_df = pd.read_csv(eda_summary_csv)
    gt_df = eda_df[eda_df["scan_id"].isin(set(test_c4["scan_id"]))].copy()
    print(f"Total ground-truth aneurysms in evaluation set: {len(gt_df)}")

    # 3. Initialize Evaluator
    evaluator = FROCEvaluator(hit_distance_mm=6.0, benchmark_fps=[0.5, 1.0, 2.0, 4.0])

    # 4. Run Baseline Inference
    print("\n--- Running 3D ResNet Baseline on Center 4 ---")
    baseline_detector = ScanDetector(
        model_type="baseline",
        checkpoint_path=baseline_ckpt,
        device=device,
        batch_size=16,
    )
    baseline_dets: List[Dict] = []

    for _, row in tqdm(test_c4.iterrows(), total=len(test_c4), desc="Baseline Center-4"):
        s_id = str(row["scan_id"])
        img_path = os.path.join(dataset_root, "images", f"{s_id}_0000.nii.gz")
        mask_path = os.path.join(dataset_root, "vessel_masks", f"{s_id}.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img_nii = nib.load(img_path)
        img_arr = np.asarray(img_nii.dataobj, dtype=np.float32)
        vessel_nii = nib.load(mask_path)
        vessel_arr = np.asarray(vessel_nii.dataobj, dtype=np.uint8) > 0
        spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

        dets = baseline_detector.detect_scan(
            image_volume=img_arr,
            vessel_mask=vessel_arr,
            modality="ct",
            spacing=spacing,
            scan_id=s_id,
        )
        baseline_dets.extend(dets)

    baseline_results = evaluator.evaluate_detections(
        all_detections=baseline_dets,
        ground_truth_df=gt_df,
        total_scans=len(test_c4),
    )

    # 5. Run Multimodal Fusion Inference
    print("\n--- Running Dual-Branch CNN + GAT Fusion on Center 4 ---")
    fusion_detector = ScanDetector(
        model_type="fusion",
        checkpoint_path=fusion_ckpt,
        device=device,
        batch_size=16,
    )
    fusion_dets: List[Dict] = []

    for _, row in tqdm(test_c4.iterrows(), total=len(test_c4), desc="Fusion Center-4"):
        s_id = str(row["scan_id"])
        img_path = os.path.join(dataset_root, "images", f"{s_id}_0000.nii.gz")
        mask_path = os.path.join(dataset_root, "vessel_masks", f"{s_id}.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img_nii = nib.load(img_path)
        img_arr = np.asarray(img_nii.dataobj, dtype=np.float32)
        vessel_nii = nib.load(mask_path)
        vessel_arr = np.asarray(vessel_nii.dataobj, dtype=np.uint8) > 0
        spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

        dets = fusion_detector.detect_scan(
            image_volume=img_arr,
            vessel_mask=vessel_arr,
            modality="ct",
            spacing=spacing,
            scan_id=s_id,
        )
        fusion_dets.extend(dets)

    fusion_results = evaluator.evaluate_detections(
        all_detections=fusion_dets,
        ground_truth_df=gt_df,
        total_scans=len(test_c4),
    )

    # 6. Save Comparison Table and Plots
    comp_df = FROCEvaluator.format_comparison_table(baseline_results, fusion_results, benchmark_fps=[0.5, 1.0, 2.0, 4.0])
    table_csv_path = os.path.join(output_dir, "center4_generalization_table.csv")
    comp_df.to_csv(table_csv_path, index=False)
    print(f"\nGeneralization comparison table saved to {table_csv_path}")

    froc_plot_path = os.path.join(output_dir, "center4_generalization_froc.png")
    FROCEvaluator.plot_froc(baseline_results, fusion_results, save_path=froc_plot_path)

    # Save complete JSON
    out_json = {
        "dataset": "TopAneu Center-4 Japan Holdout (CTA)",
        "num_scans_evaluated": len(test_c4),
        "total_ground_truth": len(gt_df),
        "baseline": {
            "mean_benchmark_sensitivity": baseline_results["mean_benchmark_sensitivity"],
            "benchmarks": baseline_results["benchmark_sensitivities"],
            "breakdown": baseline_results["benchmark_breakdown"],
        },
        "fusion": {
            "mean_benchmark_sensitivity": fusion_results["mean_benchmark_sensitivity"],
            "benchmarks": fusion_results["benchmark_sensitivities"],
            "breakdown": fusion_results["benchmark_breakdown"],
        },
    }

    metrics_json_path = os.path.join(output_dir, "center4_generalization_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    return out_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Center Evaluation")
    parser.add_argument("--max-scans", type=int, default=None, help="Limit number of scans for fast evaluation")
    parser.add_argument("--fusion-ckpt", type=str, default=None, help="Explicit checkpoint or None for Dual-Expert auto-routing")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_cross_center_generalization(
        max_scans=args.max_scans,
        fusion_ckpt=args.fusion_ckpt,
        device=args.device,
    )
