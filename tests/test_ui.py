"""
Tests for Phase 7 Clinical UI, Visualizer, and Report Generator.
"""

import os
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.ui.slice_visualizer import SliceVisualizer
from src.ui.report_generator import ClinicalReportGenerator
from src.ui.app import build_app, PRESET_CASES


def test_window_intensity():
    """Verifies radiological windowing for MRA and CTA."""
    # Synthetic MRA slice
    mra_slice = np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
    mra_win = SliceVisualizer.window_intensity(mra_slice, modality="mr")
    assert mra_win.shape == (64, 64)
    assert mra_win.dtype == np.uint8
    assert mra_win.max() <= 255
    assert mra_win.min() >= 0

    # Synthetic CTA slice with Hounsfield Units (-100 to 1200)
    cta_slice = np.random.uniform(-100, 1200, (64, 64)).astype(np.float32)
    cta_win = SliceVisualizer.window_intensity(cta_slice, modality="ct")
    assert cta_win.shape == (64, 64)
    assert cta_win.dtype == np.uint8


def test_render_tripartite_view():
    """Verifies MPR multiplanar rendering produces a valid composite image."""
    vol = np.zeros((80, 80, 80), dtype=np.float32)
    # Put bright spherical blob at (40, 40, 40)
    vol[36:44, 36:44, 36:44] = 500.0

    img = SliceVisualizer.render_tripartite_view(
        volume=vol,
        center_voxel=(40, 40, 40),
        spacing=(0.5, 0.5, 0.5),
        diameter_mm=4.5,
        score=0.85,
        location="ACom",
        modality="mr",
    )
    assert isinstance(img, Image.Image)
    assert img.width > 500
    assert img.height > 150


def test_render_overview_mip():
    """Verifies whole-brain MIP rendering with candidate markers."""
    vol = np.random.uniform(0, 100, (80, 80, 80)).astype(np.float32)
    detections = [
        {"voxel_coord": (40, 40, 40), "aneurysm_prob": 0.88, "diameter_mm": 4.2},
        {"voxel_coord": (25, 55, 30), "aneurysm_prob": 0.65, "diameter_mm": 2.8},
    ]

    mip = SliceVisualizer.render_overview_mip(
        volume=vol,
        detections=detections,
        modality="mr",
        spacing=(0.5, 0.5, 0.5),
    )
    assert isinstance(mip, Image.Image)
    assert mip.width > 200
    assert mip.height > 200


def test_classify_anatomical_location():
    """Verifies physical coordinate to anatomical location mapping."""
    loc_acom = ClinicalReportGenerator.classify_anatomical_location((95.0, 120.0, 100.0))
    assert "ACom" in loc_acom

    loc_mca_r = ClinicalReportGenerator.classify_anatomical_location((60.0, 100.0, 100.0))
    assert "Right" in loc_mca_r and "MCA" in loc_mca_r

    loc_mca_l = ClinicalReportGenerator.classify_anatomical_location((140.0, 100.0, 100.0))
    assert "Left" in loc_mca_l and "MCA" in loc_mca_l

    loc_basilar = ClinicalReportGenerator.classify_anatomical_location((100.0, 80.0, 80.0))
    assert "Basilar" in loc_basilar


def test_calculate_rupture_risk():
    """Verifies PHASES score risk tier stratification."""
    risk_very_small = ClinicalReportGenerator.calculate_rupture_risk(2.2, "MCA")
    assert risk_very_small["category"] == "Low"

    risk_small_low = ClinicalReportGenerator.calculate_rupture_risk(4.0, "MCA")
    assert risk_small_low["category"] == "Moderate"

    risk_small_high = ClinicalReportGenerator.calculate_rupture_risk(4.0, "ACom")
    assert risk_small_high["category"] == "Moderate-High"

    risk_high = ClinicalReportGenerator.calculate_rupture_risk(6.2, "ICA")
    assert risk_high["category"] == "High"

    risk_critical = ClinicalReportGenerator.calculate_rupture_risk(8.5, "Basilar")
    assert risk_critical["category"] == "Critical"


def test_generate_findings_table_and_html():
    """Verifies generation of structured dataframe and printable HTML report."""
    detections = [
        {
            "voxel_coord": (195.6, 123.3, 138.5),
            "refined_phys_coord": (85.4, 53.8, 65.1),
            "aneurysm_prob": 0.7879,
            "diameter_mm": 4.12,
            "location": "ACom",
        }
    ]

    df = ClinicalReportGenerator.generate_findings_table(detections)
    assert len(df) == 1
    assert df.iloc[0]["ID"] == "#1"
    assert "78.8%" in df.iloc[0]["Confidence"]
    assert "4.12" in df.iloc[0]["Diameter (mm)"]

    html = ClinicalReportGenerator.generate_html_report(
        scan_id="test_patient_001",
        modality="mr",
        detections=detections,
        duration_sec=14.2,
    )
    assert "test_patient_001" in html
    assert "CerebroVascular AI Diagnostic Summary" in html
    assert "4.12 mm" in html
    assert "ACom" in html


def test_preset_cases_exist():
    """Verifies all preset clinical demo scan files exist on disk."""
    for case_name, info in PRESET_CASES.items():
        assert os.path.exists(info["image"]), f"Missing image for {case_name}: {info['image']}"
        assert os.path.exists(info["mask"]), f"Missing mask for {case_name}: {info['mask']}"


def test_build_app_structure():
    """Verifies Gradio app layout compiles cleanly."""
    demo = build_app()
    assert demo is not None


def test_clinical_verdict_evaluation():
    """Verifies clinical decision tiering under standard screening."""
    from src.ui.app import analyze_cerebral_scan
    healthy_key = "🛡️ Case 4: Healthy Control TOF-MRA (Center 1, Patient _005)"
    res = analyze_cerebral_scan(demo_case_key=healthy_key)
    status_md = res[0]
    # Under uniform threshold, healthy control yields either clean negative or indeterminate review (never unequivocal positive)
    assert ("CLINICAL VERDICT: NEGATIVE" in status_md) or ("CLINICAL VERDICT: INDETERMINATE" in status_md)
    assert "CLINICAL VERDICT: POSITIVE" not in status_md

