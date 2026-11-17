"""
Clinical Free-Response Receiver Operating Characteristic (FROC) Evaluator.

Evaluates unconstrained full-volume 3D lesion detection against ground-truth aneurysm
locations. Measures detection sensitivity across benchmark false-positive rates
(0.5, 1.0, 2.0, 3.0, 4.0 FPs/scan), computes crucial lesion-size-stratified breakdowns
(<3mm, 3-5mm, <=5mm, >5mm), and provides comparative benchmarking between
pure 3D CNN baselines and topology-aware GAT fusion models.
"""

from __future__ import annotations

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch

from src.inference.candidate_proposer import CandidateProposer
from src.inference.nms3d import NMS3D
from src.data.preprocessor import VolumetricPreprocessor
from src.topology.subgraph_extractor import VesselSubgraphExtractor


class FROCEvaluator:
    """
    Computes clinical FROC curves and size-stratified sensitivity benchmarks.
    """

    BENCHMARK_FPS = [0.5, 1.0, 2.0, 3.0, 4.0]

    def __init__(
        self,
        hit_distance_mm: float = 6.0,
        benchmark_fps: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            hit_distance_mm: Maximum physical Euclidean distance (in mm) between
                a predicted centroid and a ground-truth centroid to count as a hit.
            benchmark_fps: List of false-positive-per-scan rates for sensitivity reporting.
        """
        self.hit_distance_mm = hit_distance_mm
        self.benchmark_fps = benchmark_fps or self.BENCHMARK_FPS

    def evaluate_detections(
        self,
        all_detections: List[Dict],
        ground_truth_df: pd.DataFrame,
        total_scans: int,
        score_key: str = "aneurysm_prob",
        coord_key: str = "refined_phys_coord",
    ) -> Dict:
        """
        Calculates full FROC curve and size-stratified sensitivity metrics.

        Args:
            all_detections: List of detection dictionaries across all evaluated scans.
            ground_truth_df: DataFrame of true aneurysms with columns:
                ['scan_id', 'centroid_voxel_x', 'centroid_voxel_y', 'centroid_voxel_z',
                 'spacing_x', 'spacing_y', 'spacing_z', 'max_diameter_mm']
            total_scans: Total number of scans in the evaluation cohort (including normals).
            score_key: Dictionary key containing confidence score / probability.
            coord_key: Key containing (px, py, pz) physical coordinates in mm.

        Returns:
            Dictionary containing FROC points, benchmark sensitivities, and size breakdowns.
        """
        # 1. Parse ground-truth aneurysms and their physical coordinates
        gt_by_scan: Dict[str, List[Dict]] = {}
        for _, row in ground_truth_df.iterrows():
            s_id = str(row["scan_id"])
            sx = float(row.get("spacing_x", 0.5))
            sy = float(row.get("spacing_y", 0.5))
            sz = float(row.get("spacing_z", 0.5))

            # Prefer explicit physical coordinates if present, else multiply voxel * spacing
            if "phys_x" in row:
                px, py, pz = float(row["phys_x"]), float(row["phys_y"]), float(row["phys_z"])
            else:
                px = float(row["centroid_voxel_x"]) * sx
                py = float(row["centroid_voxel_y"]) * sy
                pz = float(row["centroid_voxel_z"]) * sz

            diam = float(row.get("max_diameter_mm", 4.0))

            # Categorize size
            if diam < 3.0:
                cat = "very_small"
            elif diam <= 5.0:
                cat = "small"
            else:
                cat = "medium_large"

            if s_id not in gt_by_scan:
                gt_by_scan[s_id] = []

            gt_by_scan[s_id].append({
                "phys_coord": (px, py, pz),
                "diameter_mm": diam,
                "category": cat,
                "hit": False,
            })

        total_gt = sum(len(lesions) for lesions in gt_by_scan.values())
        gt_very_small = sum(sum(1 for l in lesions if l["category"] == "very_small") for lesions in gt_by_scan.values())
        gt_small = sum(sum(1 for l in lesions if l["category"] == "small") for lesions in gt_by_scan.values())
        gt_overall_small = gt_very_small + gt_small
        gt_medium_large = sum(sum(1 for l in lesions if l["category"] == "medium_large") for lesions in gt_by_scan.values())

        if total_gt == 0:
            return {
                "error": "No ground truth aneurysms present in evaluation set.",
                "total_scans": total_scans,
            }

        # 2. Group detections by scan
        dets_by_scan: Dict[str, List[Dict]] = {}
        for d in all_detections:
            s_id = str(d["scan_id"])
            if s_id not in dets_by_scan:
                dets_by_scan[s_id] = []
            dets_by_scan[s_id].append(d)

        # 3. Sweep score thresholds
        scores = [float(d[score_key]) for d in all_detections]
        if not scores:
            scores = [0.0]

        unique_scores = sorted(list(set(scores)), reverse=True)
        # Sample up to 100 thresholds for smooth FROC curve
        if len(unique_scores) > 100:
            thresholds = np.percentile(unique_scores, np.linspace(100, 0, 100))
        else:
            thresholds = np.array(unique_scores + [0.0])

        froc_curve: List[Dict[str, float]] = []

        for tau in thresholds:
            total_hits = 0
            hits_very_small = 0
            hits_small = 0
            hits_medium_large = 0
            total_fps = 0

            # Evaluate each scan independently at threshold tau
            for s_id in set(list(gt_by_scan.keys()) + list(dets_by_scan.keys())):
                scan_gts = [dict(g) for g in gt_by_scan.get(s_id, [])]
                scan_dets = [
                    d for d in dets_by_scan.get(s_id, [])
                    if float(d[score_key]) >= tau
                ]
                # Sort descending by score
                scan_dets = sorted(scan_dets, key=lambda x: float(x[score_key]), reverse=True)

                for det in scan_dets:
                    d_coord = np.array(det[coord_key], dtype=np.float32)

                    # Check hit to unhit GT
                    matched_gt_idx = None
                    min_dist = float("inf")

                    for g_idx, gt in enumerate(scan_gts):
                        if gt["hit"]:
                            continue
                        g_coord = np.array(gt["phys_coord"], dtype=np.float32)
                        dist = float(np.linalg.norm(d_coord - g_coord))
                        if dist <= self.hit_distance_mm and dist < min_dist:
                            min_dist = dist
                            matched_gt_idx = g_idx

                    if matched_gt_idx is not None:
                        # True Positive
                        scan_gts[matched_gt_idx]["hit"] = True
                        total_hits += 1
                        cat = scan_gts[matched_gt_idx]["category"]
                        if cat == "very_small":
                            hits_very_small += 1
                        elif cat == "small":
                            hits_small += 1
                        else:
                            hits_medium_large += 1
                    else:
                        # Check if within hit distance of an already-hit GT (duplicate)
                        is_duplicate = False
                        for gt in scan_gts:
                            if gt["hit"]:
                                g_coord = np.array(gt["phys_coord"], dtype=np.float32)
                                if np.linalg.norm(d_coord - g_coord) <= self.hit_distance_mm:
                                    is_duplicate = True
                                    break
                        if not is_duplicate:
                            total_fps += 1

            fp_per_scan = total_fps / max(total_scans, 1)
            sens_overall = total_hits / total_gt
            sens_very_small = hits_very_small / max(gt_very_small, 1)
            sens_small = hits_small / max(gt_small, 1)
            sens_overall_small = (hits_very_small + hits_small) / max(gt_overall_small, 1)
            sens_medium_large = hits_medium_large / max(gt_medium_large, 1)

            froc_curve.append({
                "threshold": float(tau),
                "fp_per_scan": float(fp_per_scan),
                "sensitivity": float(sens_overall),
                "sens_very_small": float(sens_very_small),
                "sens_small": float(sens_small),
                "sens_overall_small": float(sens_overall_small),
                "sens_medium_large": float(sens_medium_large),
                "total_fps": int(total_fps),
                "total_hits": int(total_hits),
            })

        # Ensure curve is monotonic with respect to FP rate
        froc_curve = sorted(froc_curve, key=lambda p: p["fp_per_scan"])

        # 4. Interpolate sensitivities at benchmark FP rates
        fp_vals = np.array([p["fp_per_scan"] for p in froc_curve])
        sens_vals = np.array([p["sensitivity"] for p in froc_curve])
        sens_vs_vals = np.array([p["sens_very_small"] for p in froc_curve])
        sens_s_vals = np.array([p["sens_small"] for p in froc_curve])
        sens_os_vals = np.array([p["sens_overall_small"] for p in froc_curve])
        sens_ml_vals = np.array([p["sens_medium_large"] for p in froc_curve])

        benchmark_sensitivities: Dict[float, float] = {}
        benchmark_breakdown: Dict[float, Dict[str, float]] = {}

        for b_fp in self.benchmark_fps:
            # Linear interpolation bounded by observed min and max
            sens_interp = float(np.interp(b_fp, fp_vals, sens_vals, left=0.0, right=sens_vals[-1] if len(sens_vals) else 0.0))
            sens_vs_interp = float(np.interp(b_fp, fp_vals, sens_vs_vals, left=0.0, right=sens_vs_vals[-1] if len(sens_vs_vals) else 0.0))
            sens_s_interp = float(np.interp(b_fp, fp_vals, sens_s_vals, left=0.0, right=sens_s_vals[-1] if len(sens_s_vals) else 0.0))
            sens_os_interp = float(np.interp(b_fp, fp_vals, sens_os_vals, left=0.0, right=sens_os_vals[-1] if len(sens_os_vals) else 0.0))
            sens_ml_interp = float(np.interp(b_fp, fp_vals, sens_ml_vals, left=0.0, right=sens_ml_vals[-1] if len(sens_ml_vals) else 0.0))

            benchmark_sensitivities[b_fp] = sens_interp
            benchmark_breakdown[b_fp] = {
                "overall": sens_interp,
                "very_small_under_3mm": sens_vs_interp,
                "small_3_to_5mm": sens_s_interp,
                "overall_small_le_5mm": sens_os_interp,
                "medium_large_gt_5mm": sens_ml_interp,
            }

        mean_benchmark_sens = float(np.mean(list(benchmark_sensitivities.values())))

        return {
            "total_scans": total_scans,
            "total_ground_truth": total_gt,
            "gt_counts": {
                "very_small_under_3mm": gt_very_small,
                "small_3_to_5mm": gt_small,
                "overall_small_le_5mm": gt_overall_small,
                "medium_large_gt_5mm": gt_medium_large,
            },
            "mean_benchmark_sensitivity": mean_benchmark_sens,
            "benchmark_sensitivities": benchmark_sensitivities,
            "benchmark_breakdown": benchmark_breakdown,
            "froc_curve": froc_curve,
        }

    @staticmethod
    def format_comparison_table(
        baseline_eval: Dict,
        proposed_eval: Dict,
        benchmark_fps: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Creates a clean publication-ready comparison table comparing baseline vs proposed.
        """
        fps = benchmark_fps or [0.5, 1.0, 2.0, 3.0, 4.0]
        rows = []

        for fp in fps:
            b_val = baseline_eval["benchmark_sensitivities"].get(fp, 0.0) * 100.0
            p_val = proposed_eval["benchmark_sensitivities"].get(fp, 0.0) * 100.0
            diff = p_val - b_val

            # Small lesion breakdown at this FP rate
            b_small = baseline_eval["benchmark_breakdown"][fp]["overall_small_le_5mm"] * 100.0
            p_small = proposed_eval["benchmark_breakdown"][fp]["overall_small_le_5mm"] * 100.0
            diff_small = p_small - b_small

            rows.append({
                "FP / Scan": f"{fp:.1f}",
                "Baseline Sens (All)": f"{b_val:.1f}%",
                "Fusion Sens (All)": f"{p_val:.1f}%",
                "Net Delta (All)": f"{diff:+.1f}%",
                "Baseline Sens (<=5mm)": f"{b_small:.1f}%",
                "Fusion Sens (<=5mm)": f"{p_small:.1f}%",
                "Net Delta (<=5mm)": f"{diff_small:+.1f}%",
            })

        # Add Mean Row
        b_mean = baseline_eval["mean_benchmark_sensitivity"] * 100.0
        p_mean = proposed_eval["mean_benchmark_sensitivity"] * 100.0
        rows.append({
            "FP / Scan": "Mean Score",
            "Baseline Sens (All)": f"{b_mean:.1f}%",
            "Fusion Sens (All)": f"{p_mean:.1f}%",
            "Net Delta (All)": f"{p_mean - b_mean:+.1f}%",
            "Baseline Sens (<=5mm)": "-",
            "Fusion Sens (<=5mm)": "-",
            "Net Delta (<=5mm)": "-",
        })

        return pd.DataFrame(rows)

    @staticmethod
    def plot_froc(
        baseline_eval: Dict,
        proposed_eval: Dict,
        save_path: str = "artifacts/froc_comparison_curve.png",
        max_fp: float = 4.0,
    ) -> None:
        """
        Plots publication-quality FROC curves for Baseline vs Fusion model.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- Panel 1: Overall Aneurysm Sensitivity ---
        b_fps = [p["fp_per_scan"] for p in baseline_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]
        b_sens = [p["sensitivity"] * 100.0 for p in baseline_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]

        p_fps = [p["fp_per_scan"] for p in proposed_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]
        p_sens = [p["sensitivity"] * 100.0 for p in proposed_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]

        ax1.plot(b_fps, b_sens, color="#2563EB", linewidth=2.5, label=f"3D ResNet Baseline (Mean: {baseline_eval['mean_benchmark_sensitivity']*100:.1f}%)")
        ax1.plot(p_fps, p_sens, color="#10B981", linewidth=2.5, label=f"Dual-Branch Fusion (Mean: {proposed_eval['mean_benchmark_sensitivity']*100:.1f}%)")

        for b_fp in [0.5, 1.0, 2.0, 4.0]:
            ax1.axvline(x=b_fp, color="#94A3B8", linestyle="--", alpha=0.5)

        ax1.set_xlabel("False Positives per Scan (FPs/Scan)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Detection Sensitivity (%)", fontsize=12, fontweight="bold")
        ax1.set_title("Full-Scan Clinical FROC: All Aneurysms", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlim(0, max_fp)
        ax1.set_ylim(0, 105)
        ax1.legend(loc="lower right", fontsize=11, frameon=True)

        # --- Panel 2: Small Aneurysms (<= 5mm) Breakdown ---
        b_s_sens = [p["sens_overall_small"] * 100.0 for p in baseline_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]
        p_s_sens = [p["sens_overall_small"] * 100.0 for p in proposed_eval["froc_curve"] if p["fp_per_scan"] <= max_fp]

        ax2.plot(b_fps, b_s_sens, color="#DC2626", linestyle="-.", linewidth=2.5, label="Baseline: Small (<=5mm)")
        ax2.plot(p_fps, p_s_sens, color="#059669", linewidth=2.5, label="Fusion: Small (<=5mm)")

        for b_fp in [0.5, 1.0, 2.0, 4.0]:
            ax2.axvline(x=b_fp, color="#94A3B8", linestyle="--", alpha=0.5)

        ax2.set_xlabel("False Positives per Scan (FPs/Scan)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Small Lesion Sensitivity (%)", fontsize=12, fontweight="bold")
        ax2.set_title("Clinical FROC Breakdown: Small Aneurysms (<= 5mm)", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xlim(0, max_fp)
        ax2.set_ylim(0, 105)
        ax2.legend(loc="lower right", fontsize=11, frameon=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"FROC curve figure saved to {save_path}")


def run_froc_benchmark(
    split_csv: str = "configs/val_split.csv",
    eda_summary_csv: str = "artifacts/eda/aneurysm_annotations_summary.csv",
    dataset_root: str = "dataset(topAneu)",
    baseline_ckpt: str = "checkpoints/best_baseline_resnet3d.pt",
    fusion_ckpt: Optional[str] = None,
    output_dir: str = "artifacts/froc_eval",
    max_scans: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Executes full clinical detection and FROC evaluation on a designated split.
    """
    from src.inference.scan_detector import ScanDetector
    import nibabel as nib
    from tqdm import tqdm
    import json

    os.makedirs(output_dir, exist_ok=True)
    split_df = pd.read_csv(split_csv)
    if max_scans is not None:
        split_df = split_df.head(max_scans)

    print(f"Running clinical FROC benchmark on {len(split_df)} scans from {split_csv}...")

    # Load ground truth
    eda_df = pd.read_csv(eda_summary_csv)
    gt_df = eda_df[eda_df["scan_id"].isin(set(split_df["scan_id"]))].copy()
    print(f"Loaded {len(gt_df)} ground-truth aneurysms for evaluation cohort.")

    evaluator = FROCEvaluator(hit_distance_mm=6.0, benchmark_fps=[0.5, 1.0, 2.0, 3.0, 4.0])

    # 1. Evaluate Baseline Model
    print("\n--- Evaluating Phase 3 3D CNN Baseline ---")
    baseline_detector = ScanDetector(
        model_type="baseline",
        checkpoint_path=baseline_ckpt,
        device=device,
        batch_size=16,
    )
    baseline_dets: List[Dict] = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Baseline Evaluation"):
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
            modality=str(row["modality"]),
            spacing=spacing,
            scan_id=s_id,
        )
        baseline_dets.extend(dets)

    baseline_results = evaluator.evaluate_detections(
        all_detections=baseline_dets,
        ground_truth_df=gt_df,
        total_scans=len(split_df),
    )

    # 2. Evaluate Multimodal Dual-Branch Fusion Model
    print("\n--- Evaluating Phase 5 Multimodal Dual-Branch Fusion Model ---")
    fusion_detector = ScanDetector(
        model_type="fusion",
        checkpoint_path=fusion_ckpt,
        device=device,
        batch_size=16,
    )
    fusion_dets: List[Dict] = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Fusion Evaluation"):
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
            modality=str(row["modality"]),
            spacing=spacing,
            scan_id=s_id,
        )
        fusion_dets.extend(dets)

    fusion_results = evaluator.evaluate_detections(
        all_detections=fusion_dets,
        ground_truth_df=gt_df,
        total_scans=len(split_df),
    )

    # 3. Format Comparison Table and Plots
    comp_df = FROCEvaluator.format_comparison_table(baseline_results, fusion_results)
    table_csv = os.path.join(output_dir, "froc_comparison_table.csv")
    comp_df.to_csv(table_csv, index=False)
    print("\n=================== CLINICAL FROC BENCHMARK ===================")
    print(comp_df.to_string(index=False))
    print("===============================================================\n")

    plot_path = os.path.join(output_dir, "froc_comparison_curve.png")
    FROCEvaluator.plot_froc(baseline_results, fusion_results, save_path=plot_path)

    # Save summary JSON
    out_data = {
        "split_csv": split_csv,
        "scans_evaluated": len(split_df),
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
    with open(os.path.join(output_dir, "froc_summary.json"), "w") as f:
        json.dump(out_data, f, indent=2)

    return out_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clinical FROC Detection Benchmark")
    parser.add_argument("--split-csv", type=str, default="configs/val_split.csv")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--fusion-ckpt", type=str, default=None, help="Explicit checkpoint or None for Dual-Expert auto-routing")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_froc_benchmark(
        split_csv=args.split_csv,
        max_scans=args.max_scans,
        fusion_ckpt=args.fusion_ckpt,
        device=args.device,
    )

