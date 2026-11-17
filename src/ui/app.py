"""
Phase 7: Interactive Clinical Decision Support Web UI & Diagnostic Report Generator.

Powered by Gradio, this clinical workstation connects the Dual-Expert Multimodal
Neural Network (3D CNN + Arterial Topology GATv2) to provide automated intracranial
aneurysm detection, multiplanar orthogonal visualization (MPR), and PHASES rupture
risk stratification.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gradio as gr
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from src.inference.scan_detector import ScanDetector
from src.ui.slice_visualizer import SliceVisualizer
from src.ui.report_generator import ClinicalReportGenerator


# Preset clinical cases for rapid demonstration
PRESET_CASES = {
    "🎯 Case 1: TOF-MRA with Aneurysm (Center 1, Patient _193)": {
        "scan_id": "topaneu_center1_mr_193",
        "image": "dataset(topAneu)/images/topaneu_center1_mr_193_0000.nii.gz",
        "mask": "dataset(topAneu)/vessel_masks/topaneu_center1_mr_193.nii.gz",
        "modality": "mr",
        "notes": "Positive TOF-MRA screening scan with high-confidence bifurcational aneurysm.",
    },
    "🎯 Case 2: TOF-MRA with Aneurysm (Center 1, Patient _017)": {
        "scan_id": "topaneu_center1_mr_017",
        "image": "dataset(topAneu)/images/topaneu_center1_mr_017_0000.nii.gz",
        "mask": "dataset(topAneu)/vessel_masks/topaneu_center1_mr_017.nii.gz",
        "modality": "mr",
        "notes": "Positive TOF-MRA scan with ACom arterial outpouching.",
    },
    "🎯 Case 3: CTA Zero-Shot Holdout (Center 4 Japan, Patient _002)": {
        "scan_id": "topaneu_center4_ct_002",
        "image": "dataset(topAneu)/images/topaneu_center4_ct_002_0000.nii.gz",
        "mask": "dataset(topAneu)/vessel_masks/topaneu_center4_ct_002.nii.gz",
        "modality": "ct",
        "notes": "Emergency department CTA with skull-base bone and iodinated contrast.",
    },
    "🛡️ Case 4: Healthy Control TOF-MRA (Center 1, Patient _005)": {
        "scan_id": "topaneu_center1_mr_005",
        "image": "dataset(topAneu)/images/topaneu_center1_mr_005_0000.nii.gz",
        "mask": "dataset(topAneu)/vessel_masks/topaneu_center1_mr_005.nii.gz",
        "modality": "mr",
        "notes": "Normal vascular anatomy with zero aneurysms (negative control).",
    },
}

# Cache for singletons to avoid re-loading weights on each click
DETECTOR_CACHE: Dict[str, ScanDetector] = {}


def get_detector(model_type: str, explicit_ckpt: Optional[str] = None) -> ScanDetector:
    """Retrieves or instantiates cached ScanDetector."""
    cache_key = f"{model_type}_{explicit_ckpt}"
    if cache_key not in DETECTOR_CACHE:
        DETECTOR_CACHE[cache_key] = ScanDetector(
            model_type=model_type,
            checkpoint_path=explicit_ckpt,
            mra_checkpoint="checkpoints/best_fusion_mra.pt",
            cta_checkpoint="checkpoints/best_fusion_cta.pt",
        )
    return DETECTOR_CACHE[cache_key]


def analyze_cerebral_scan(
    demo_case_key: str,
    custom_img_file: Optional[Any] = None,
    custom_mask_file: Optional[Any] = None,
    modality_choice: str = "Auto-Detect",
    model_choice: str = "Dual-Expert Mode (Auto-Route)",
    clinical_mode: str = "🛡️ Standard Diagnostic Screening (High Specificity - Zero Clutter)",
    conf_threshold: float = 0.50,
    nms_distance_mm: float = 6.0,
    progress=gr.Progress(),
) -> Tuple:
    """
    Main diagnostic execution pipeline.
    """
    start_time = time.time()
    os.makedirs("artifacts/reports", exist_ok=True)

    # 1. Resolve Input File Paths & Operating Presets
    is_preset = demo_case_key != "📂 Use Custom Upload Below"

    # Resolve Clinical Operating Thresholds (Universal across ALL scans - presets & custom uploads)
    if "Standard Diagnostic" in clinical_mode:
        effective_conf = 0.78  # High-specificity screening: eliminates normal bifurcation false alarms
        effective_nms = 6.0
    elif "Emergency" in clinical_mode:
        effective_conf = 0.65  # High-sensitivity ER triage: catches subtle/smaller aneurysms
        effective_nms = 5.0
    else:
        # Research & Engineering Mode (Manual Sliders)
        effective_conf = conf_threshold
        effective_nms = nms_distance_mm

    if is_preset:
        case_info = PRESET_CASES[demo_case_key]
        img_path = case_info["image"]
        mask_path = case_info["mask"]
        scan_id = case_info["scan_id"]
        detected_modality = case_info["modality"]
    else:
        if custom_img_file is None:
            raise gr.Error("Please upload a 3D NIfTI scan (.nii or .nii.gz) or select a preset case.")
        img_path = custom_img_file.name if hasattr(custom_img_file, "name") else str(custom_img_file)
        scan_id = os.path.basename(img_path).replace(".nii.gz", "").replace(".nii", "")
        mask_path = (custom_mask_file.name if hasattr(custom_mask_file, "name") else str(custom_mask_file)) if custom_mask_file is not None else None
        detected_modality = "ct" if ("ct" in scan_id.lower()) else "mr"

    if modality_choice == "TOF-MRA (MR)":
        final_modality = "mr"
    elif modality_choice == "CTA (CT)":
        final_modality = "ct"
    else:
        final_modality = detected_modality

    progress(0.1, desc="Loading 3D volumetric NIfTI headers...")
    img_nii = nib.load(img_path)
    img_arr = np.asarray(img_nii.dataobj, dtype=np.float32)
    spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

    # 2. Resolve Vessel Mask
    if mask_path and os.path.exists(mask_path):
        vessel_nii = nib.load(mask_path)
        vessel_arr = np.asarray(vessel_nii.dataobj, dtype=np.uint8) > 0
    else:
        progress(0.2, desc="Executing Automated Intracranial Skull-Stripping & Artery Extraction...")
        from src.data.preprocessor import VolumetricPreprocessor
        prep = VolumetricPreprocessor()
        brain_mask = prep.extract_brain_mask(img_arr, spacing=spacing, modality=final_modality)
        vessel_arr = prep.extract_automated_vessels(img_arr, brain_mask=brain_mask, spacing=spacing, modality=final_modality)

    # 3. Configure Model Engine
    progress(0.3, desc=f"Initializing {model_choice} engine...")
    if "Baseline" in model_choice:
        detector = get_detector(model_type="baseline", explicit_ckpt="checkpoints/best_baseline_resnet3d.pt")
    elif "MRA Specialist" in model_choice:
        detector = get_detector(model_type="fusion", explicit_ckpt="checkpoints/best_fusion_mra.pt")
    elif "CTA Specialist" in model_choice:
        detector = get_detector(model_type="fusion", explicit_ckpt="checkpoints/best_fusion_cta.pt")
    else:
        # Dual-Expert Auto-Routing
        detector = get_detector(model_type="fusion", explicit_ckpt=None)

    detector.conf_threshold = effective_conf
    detector.nms_distance_mm = effective_nms
    detector.nms.distance_threshold_mm = effective_nms
    detector.nms.score_threshold = effective_conf

    # 4. Run Detection Pipeline
    progress(0.5, desc="Extracting arterial topology & evaluating candidates...")
    detections = detector.detect_scan(
        image_volume=img_arr,
        vessel_mask=vessel_arr,
        modality=final_modality,
        spacing=spacing,
        scan_id=scan_id,
    )
    duration = time.time() - start_time

    progress(0.85, desc="Rendering multiplanar views & generating clinical report...")
    # 5. Format Findings Table & Report
    findings_df = ClinicalReportGenerator.generate_findings_table(detections, spacing=spacing)
    html_report = ClinicalReportGenerator.generate_html_report(
        scan_id=scan_id,
        modality=final_modality,
        detections=detections,
        spacing=spacing,
        duration_sec=duration,
    )

    # Save exports
    html_path = os.path.join("artifacts/reports", f"report_{scan_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    csv_path = os.path.join("artifacts/reports", f"findings_{scan_id}.csv")
    findings_df.to_csv(csv_path, index=False)

    # 6. Render Visualizations
    mip_img = SliceVisualizer.render_overview_mip(
        volume=img_arr,
        detections=detections,
        modality=final_modality,
        spacing=spacing,
    )

    if detections:
        first_det = detections[0]
        v_coord = first_det.get("refined_voxel_coord") or first_det.get("voxel_coord")
        if v_coord is None:
            phys = first_det.get("refined_phys_coord", (0.0, 0.0, 0.0))
            v_coord = (phys[0] / max(spacing[0], 1e-4), phys[1] / max(spacing[1], 1e-4), phys[2] / max(spacing[2], 1e-4))
        mpr_img = SliceVisualizer.render_tripartite_view(
            volume=img_arr,
            center_voxel=v_coord,
            spacing=spacing,
            diameter_mm=first_det.get("diameter_mm"),
            score=first_det.get("aneurysm_prob"),
            location=first_det.get("location") or ClinicalReportGenerator.classify_anatomical_location(first_det.get("refined_phys_coord"), spacing),
            modality=final_modality,
        )
        lesion_choices = [
            f"Lesion #{i+1} - {d.get('aneurysm_prob', 0)*100:.1f}% ({d.get('diameter_mm', 0):.1f}mm)"
            for i, d in enumerate(detections)
        ]
        selected_choice = lesion_choices[0]
    else:
        # Fallback view centered in brain volume
        mid_voxel = (img_arr.shape[0] // 2, img_arr.shape[1] // 2, img_arr.shape[2] // 2)
        mpr_img = SliceVisualizer.render_tripartite_view(
            volume=img_arr,
            center_voxel=mid_voxel,
            spacing=spacing,
            diameter_mm=0.0,
            score=0.0,
            location="Center of Brain (No Lesions)",
            modality=final_modality,
        )
        lesion_choices = ["No lesions detected"]
        selected_choice = "No lesions detected"

    # Count clinically actionable aneurysms (confidence >= 0.50)
    actionable_count = sum(float(d.get("aneurysm_prob", 0.0)) >= 0.50 for d in detections)
    max_conf = max([float(d.get("aneurysm_prob", 0.0)) for d in detections], default=0.0)
    risk_summary = findings_df["Rupture Risk"].tolist() if not findings_df.empty else []
    highest_risk = "Normal / Negative Scan"
    for r in ["Critical", "High", "Moderate-High", "Moderate", "Low", "Borderline / Low Suspicion"]:
        if r in risk_summary:
            highest_risk = r
            break
    if highest_risk == "Normal / Negative Scan" and actionable_count > 0:
        highest_risk = "Low / Surveillance"

    if len(detections) == 0 or actionable_count == 0:
        verdict_badge = """
<div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 32px;">✅</span>
        <div>
            <div style="color: #15803d; font-size: 18px; font-weight: 800; letter-spacing: -0.2px;">
                CLINICAL VERDICT: NEGATIVE (No Aneurysms Detected)
            </div>
            <div style="color: #166534; font-size: 13px; margin-top: 3px;">
                <strong>Screening Evaluation:</strong> Systematically analyzed all vascular bifurcations across the Circle of Willis. Zero actionable lesions detected above the diagnostic threshold. Standard non-aneurysmal screening scan.
            </div>
        </div>
    </div>
</div>
"""
    elif (actionable_count == 1 and max_conf < 0.85 and highest_risk not in ["Critical", "High"]):
        verdict_badge = f"""
<div style="background: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 32px;">🟡</span>
        <div>
            <div style="color: #b45309; font-size: 18px; font-weight: 800; letter-spacing: -0.2px;">
                CLINICAL VERDICT: INDETERMINATE / BORDERLINE CANDIDATE (Review Normal Variant)
            </div>
            <div style="color: #92400e; font-size: 13px; margin-top: 3px;">
                <strong>Clinical Note:</strong> 1 borderline candidate identified at {max_conf*100:.1f}% confidence. Likely normal anatomical arterial bifurcation or vessel branching geometry. Correlate with multiplanar orthogonal slices below to verify non-aneurysmal tubular lumen.
            </div>
        </div>
    </div>
</div>
"""
    else:
        verdict_badge = f"""
<div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 32px;">⚠️</span>
        <div>
            <div style="color: #b91c1c; font-size: 18px; font-weight: 800; letter-spacing: -0.2px;">
                CLINICAL VERDICT: POSITIVE ({actionable_count} Aneurysm Candidate(s) Detected)
            </div>
            <div style="color: #7f1d1d; font-size: 13px; margin-top: 3px;">
                <strong>Action Alert:</strong> Focal vascular outpouching identified with highest rupture risk tier: <strong>{highest_risk}</strong>. Target multiplanar orthogonal slices rendered below for radiological confirmation.
            </div>
        </div>
    </div>
</div>
"""

    status_md = verdict_badge + f"""
### 🩺 Patient Scan Metadata
* **Patient / Scan ID**: `{scan_id}`
* **Modality Detected**: `{final_modality.upper()} Angiography`
* **Clinical Mode**: `{clinical_mode.split('(')[0].strip()}`
* **Model Engine**: `{model_choice}`
* **Actionable Aneurysms**: **`{actionable_count}`**
* **Inference Runtime**: **`{duration:.2f} seconds`**
"""

    state_data = {
        "volume": img_arr,
        "spacing": spacing,
        "modality": final_modality,
        "detections": detections,
    }

    progress(1.0, desc="Diagnostic analysis complete.")
    return (
        status_md,
        findings_df,
        mpr_img,
        mip_img,
        gr.update(choices=lesion_choices, value=selected_choice),
        html_path,
        csv_path,
        state_data,
    )


def switch_lesion_view(selected_lesion_str: str, state_data: dict) -> Image.Image:
    """Updates the Tripartite MPR slice view when clinician selects a different lesion."""
    if not state_data or not state_data.get("detections"):
        return None

    detections = state_data["detections"]
    volume = state_data["volume"]
    spacing = state_data["spacing"]
    modality = state_data["modality"]

    # Parse index from string e.g. "Lesion #2 - 78.4% (4.2mm)"
    try:
        idx_str = selected_lesion_str.split("#")[1].split(" ")[0].strip()
        idx = int(idx_str) - 1
        det = detections[idx]
    except Exception:
        det = detections[0]

    v_coord = det.get("refined_voxel_coord") or det.get("voxel_coord")
    if v_coord is None:
        phys = det.get("refined_phys_coord", (0.0, 0.0, 0.0))
        v_coord = (phys[0] / max(spacing[0], 1e-4), phys[1] / max(spacing[1], 1e-4), phys[2] / max(spacing[2], 1e-4))

    return SliceVisualizer.render_tripartite_view(
        volume=volume,
        center_voxel=v_coord,
        spacing=spacing,
        diameter_mm=det.get("diameter_mm"),
        score=det.get("aneurysm_prob"),
        location=det.get("location") or ClinicalReportGenerator.classify_anatomical_location(det.get("refined_phys_coord"), spacing),
        modality=modality,
    )


def build_app():
    """Constructs the Gradio web interface."""
    with gr.Blocks(title="CerebroVascular AI Workstation") as demo:
        # Header
        gr.Markdown(
            """
# 🧠 CerebroVascular AI: Dual-Expert Intracranial Aneurysm Workstation
### Multimodal 3D CNN + Arterial Topology Graph Attention Network (GATv2)
*Developed by Dayananda Sagar University (DSU) Major Project Research Team (2026–2027)*
---
"""
        )

        state_holder = gr.State()

        with gr.Tabs():
            # TAB 1: Clinical Workstation
            with gr.TabItem("🏥 Clinical Diagnostic Workstation"):
                with gr.Row():
                    # Left Column: Input and Parameters
                    with gr.Column(scale=4):
                        gr.Markdown("#### 📥 1. Select Patient Scan")
                        preset_dropdown = gr.Dropdown(
                            label="Preset Clinical Scans (Instant Demo)",
                            choices=list(PRESET_CASES.keys()) + ["📂 Use Custom Upload Below"],
                            value="🎯 Case 1: TOF-MRA with Aneurysm (Center 1, Patient _193)",
                        )

                        with gr.Accordion("📂 Custom NIfTI Volume Upload", open=False):
                            custom_img = gr.File(label="Patient 3D Brain Image (.nii / .nii.gz)", file_types=[".gz", ".nii"])
                            custom_mask = gr.File(label="Optional Vessel Mask (.nii / .nii.gz)", file_types=[".gz", ".nii"])

                        gr.Markdown("#### ⚙️ 2. Clinical Settings & Model Engine")
                        mode_radio = gr.Radio(
                            label="Clinical Operational Mode",
                            choices=[
                                "🛡️ Standard Diagnostic Screening (High Specificity - Zero Clutter)",
                                "🚨 High-Sensitivity Emergency Triage (ER / Trauma)",
                                "🔬 Research & Engineering Mode (Manual Sliders)",
                            ],
                            value="🛡️ Standard Diagnostic Screening (High Specificity - Zero Clutter)",
                        )
                        modality_radio = gr.Radio(
                            label="Imaging Modality",
                            choices=["Auto-Detect", "TOF-MRA (MR)", "CTA (CT)"],
                            value="Auto-Detect",
                        )
                        model_radio = gr.Radio(
                            label="Neural Architecture Engine",
                            choices=[
                                "Dual-Expert Mode (Auto-Route)",
                                "Phase 6 MRA Specialist",
                                "Phase 6 CTA Specialist",
                                "Phase 3 3D CNN Baseline",
                            ],
                            value="Dual-Expert Mode (Auto-Route)",
                        )

                        with gr.Accordion("🔬 Advanced Engineering Diagnostics (Custom Thresholds)", open=False):
                            gr.Markdown(
                                "_Note: Leave collapsed for standard hospital use. Automatically calibrated by Clinical Operational Mode above._"
                            )
                            conf_slider = gr.Slider(
                                label="Manual Confidence Cutoff (P_aneurysm)",
                                minimum=0.05,
                                maximum=0.90,
                                value=0.50,
                                step=0.05,
                            )
                            nms_slider = gr.Slider(
                                label="3D-NMS Clustering Radius (mm)",
                                minimum=3.0,
                                maximum=12.0,
                                value=6.0,
                                step=0.5,
                            )

                        analyze_btn = gr.Button("🚀 Analyze Cerebral Vasculature", variant="primary", size="lg")

                        status_output = gr.Markdown("Ready for scan analysis.")

                    # Right Column: Visualizations & Findings
                    with gr.Column(scale=8):
                        gr.Markdown("#### 🎯 Multiplanar Orthogonal Slices (Axial, Coronal, Sagittal)")
                        lesion_selector = gr.Dropdown(
                            label="Select Detected Aneurysm Target",
                            choices=["No analysis run yet"],
                            value="No analysis run yet",
                        )
                        mpr_display = gr.Image(label="Multiplanar Orthogonal View", type="pil")

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### 🧠 Whole-Brain Maximum Intensity Projection (MIP)")
                                mip_display = gr.Image(label="MIP Overview", type="pil")
                            with gr.Column(scale=1):
                                gr.Markdown("#### 📥 Export Clinical Diagnostic Reports")
                                html_download = gr.File(label="Download HTML Diagnostic Report")
                                csv_download = gr.File(label="Download Findings CSV")

                gr.Markdown("### 📋 Morphological Findings & Rupture Risk Table")
                findings_table = gr.Dataframe(
                    headers=[
                        "ID",
                        "Confidence",
                        "Diameter (mm)",
                        "Size Category",
                        "Centroid (X,Y,Z mm)",
                        "Anatomical Location",
                        "Rupture Risk",
                        "Clinical Action",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                )

            # TAB 2: Model Architecture & Benchmarks
            with gr.TabItem("📊 Empirical Benchmarks & Academic Defense"):
                gr.Markdown(
                    r"""
### 🏆 Empirical Benchmark Summary: Baseline vs. Dual-Expert Architecture

| Benchmark Metric | Phase 3 Baseline (3D CNN) | Phase 5 Unified Model | Phase 6 Dual-Expert Architecture | Clinical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Center-4 CTA Sens @ 0.5 FP/scan** | `0.0%` | `5.0%` | **`9.2%`** | Rescues low-false-alarm clinical regime |
| **Center-4 CTA Sens @ 1.0 FP/scan** | `0.0%` | `11.7%` | **`21.5%`** | **+$21.5\%$** sensitivity gain |
| **Center-4 CTA Sens @ 2.0 FP/scan** | `0.0%` | `20.0%` | **`44.6%`** | **+$44.6\%$** zero-shot transfer |
| **Center-4 CTA Sens @ 4.0 FP/scan** | `0.0%` | `26.0%` | **`72.9%`** | **Almost $3\times$ improvement** |
| **Small Lesions ($\le 5\text{ mm}$)** | `0.0%` | `12.0%` | **`85.9%`** | Breakthrough detection on bleeding-prone small aneurysms |
| **Diameter Regression Error (MAE)** | 2.58 mm | 2.14 mm | **`1.86 mm`** | Sub-2mm caliper precision |
| **Overall Validation AUROC** | 0.8120 | 0.8410 | **`0.8971` (CTA) / `0.8653` (MRA)** | Highest cross-modality discrimination |

---

### 🔬 Why Vascular Topology + Dual-Experts Matter:
1. **Disambiguation of Normal Bifurcations**: Arterial forks share visual curvature and intensity with aneurysms. The **GATv2 Graph Attention** branch evaluates local vessel branch angles and caliber ratios, suppressing false positives.
2. **Elimination of Modality Negative Transfer**: TOF-MRA measures blood flow spin physics (bone is pitch black), while CTA measures X-ray Hounsfield attenuation (skull base bone $>1000$ HU). Allocating specialized 3D visual kernels while sharing the vascular topology network maximizes performance on both modalities.
"""
                )

        # Wire Event Callbacks
        analyze_btn.click(
            fn=analyze_cerebral_scan,
            inputs=[
                preset_dropdown,
                custom_img,
                custom_mask,
                modality_radio,
                model_radio,
                mode_radio,
                conf_slider,
                nms_slider,
            ],
            outputs=[
                status_output,
                findings_table,
                mpr_display,
                mip_display,
                lesion_selector,
                html_download,
                csv_download,
                state_holder,
            ],
        )

        lesion_selector.change(
            fn=switch_lesion_view,
            inputs=[lesion_selector, state_holder],
            outputs=[mpr_display],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    # Launch locally on port 7860
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
