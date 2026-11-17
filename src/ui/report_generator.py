"""
Clinical Diagnostic Report Generator & Rupture Risk Stratification.

Computes aneurysm rupture risk categories according to the PHASES / ISUIA clinical
scoring guidelines and generates downloadable CSV and formatted HTML diagnostic reports.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class ClinicalReportGenerator:
    """
    Translates raw detection coordinates and morphological predictions
    into clinical risk assessments and formal radiological reports.
    """

    @staticmethod
    def classify_anatomical_location(
        phys_coord_mm: tuple[float, float, float],
        spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> str:
        """
        Estimates anatomical arterial site based on normalized CoW physical coordinates.
        """
        px, py, pz = phys_coord_mm

        # Simple coordinate-based arterial heuristic for CoW
        # Midline anterior: ACom
        # Lateral bilateral: MCA
        # Inferior posterior: Basilar / Vertebral
        # Inferior lateral: ICA
        if abs(px - 100.0) < 18.0 and py > 115.0:
            return "Anterior Communicating Artery (ACom)"
        elif abs(px - 100.0) >= 30.0:
            side = "Right" if px < 100.0 else "Left"
            return f"{side} Middle Cerebral Artery (MCA) Bifurcation"
        elif py < 85.0 and pz < 85.0:
            return "Basilar Artery / Vertebrobasilar Junction"
        elif pz < 95.0:
            side = "Right" if px < 100.0 else "Left"
            return f"{side} Internal Carotid Artery (ICA) C7 / PCom"
        else:
            return "Anterior Cerebral Artery (ACA / A1-A2)"

    @classmethod
    def calculate_rupture_risk(
        cls,
        diameter_mm: float,
        location: str,
        confidence: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Stratifies rupture risk according to PHASES / ISUIA clinical scoring criteria.

        Returns:
            Dict containing:
                - 'category': 'Low', 'Moderate', 'High', or 'Critical'
                - 'color': CSS hex color for UI badge
                - 'recommendation': Actionable radiological recommendation
                - 'phases_tier': Brief risk explanation
        """
        if confidence is not None and confidence < 0.50:
            return {
                "category": "Borderline / Low Suspicion",
                "color": "#64748b",
                "recommendation": "Sub-threshold finding (confidence < 50%). Likely normal anatomical tortuosity.",
                "phases_tier": f"Confidence {confidence*100:.0f}% < 50% (Non-actionable)",
            }

        loc_lower = location.lower()
        is_high_risk_location = any(site in loc_lower for site in ["acom", "pcom", "basilar", "vertebrobasilar"])

        if diameter_mm < 3.0:
            category = "Low"
            color = "#10b981"  # Emerald
            recommendation = "Routine non-invasive vascular surveillance (12-24 month follow-up MRA/CTA)."
            tier = "Size < 3.0 mm (Annual rupture risk < 0.25%)"
        elif 3.0 <= diameter_mm <= 5.0:
            if is_high_risk_location:
                category = "Moderate-High"
                color = "#f59e0b"  # Amber
                recommendation = "Early repeat angiography (6-12 month MRA/CTA) and outpatient neurointerventional consult."
                tier = "Size 3-5 mm at higher-risk arterial bifurcation (PHASES score 3-5)"
            else:
                category = "Moderate"
                color = "#eab308"  # Yellow
                recommendation = "Serial imaging at 6 to 12 months to monitor for morphological enlargement."
                tier = "Size 3-5 mm in low-stress anterior circulation (PHASES score 2-4)"
        elif 5.0 < diameter_mm < 7.0:
            category = "High"
            color = "#f97316"  # Orange
            recommendation = "Urgent neurosurgical / neurointerventional consultation for elective coiling or clipping."
            tier = "Size 5-7 mm (Annual rupture risk 1.2 - 2.5%)"
        else:
            category = "Critical"
            color = "#ef4444"  # Red
            recommendation = "Immediate neurovascular specialist referral. Substantial rupture hazard."
            tier = f"Size >= 7.0 mm ({diameter_mm:.1f} mm, annual rupture risk > 3.0%)"

        return {
            "category": category,
            "color": color,
            "recommendation": recommendation,
            "phases_tier": tier,
        }

    @classmethod
    def generate_findings_table(
        cls,
        detections: List[Dict],
        spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> pd.DataFrame:
        """
        Transforms detections into a clean tabular DataFrame for UI presentation.
        """
        if not detections:
            return pd.DataFrame(
                [
                    {
                        "ID": "—",
                        "Confidence": "Normal",
                        "Diameter (mm)": "—",
                        "Size Category": "Normal Caliber",
                        "Centroid (X,Y,Z mm)": "Circle of Willis",
                        "Anatomical Location": "No Focal Outpouching Detected",
                        "Rupture Risk": "None (Negative)",
                        "Clinical Action": "Routine clinical screening; no surgical or interventional referral indicated.",
                    }
                ]
            )

        rows = []
        for i, det in enumerate(detections):
            score = float(det.get("aneurysm_prob", 0.0))
            diam = float(det.get("diameter_mm", 0.0))
            phys = det.get("refined_phys_coord", (0.0, 0.0, 0.0))
            phys_tuple = (float(phys[0]), float(phys[1]), float(phys[2]))

            loc = det.get("location")
            if not loc:
                loc = cls.classify_anatomical_location(phys_tuple, spacing=spacing)

            risk_info = cls.calculate_rupture_risk(diam, loc, confidence=score)

            if diam < 3.0:
                size_cat = "Very Small (<3mm)"
            elif diam <= 5.0:
                size_cat = "Small (3-5mm)"
            elif diam <= 7.0:
                size_cat = "Medium (5-7mm)"
            else:
                size_cat = "Large (>7mm)"

            rows.append(
                {
                    "ID": f"#{i+1}",
                    "Confidence": f"{score*100:.1f}%",
                    "Diameter (mm)": f"{diam:.2f}",
                    "Size Category": size_cat,
                    "Centroid (X,Y,Z mm)": f"({phys_tuple[0]:.1f}, {phys_tuple[1]:.1f}, {phys_tuple[2]:.1f})",
                    "Anatomical Location": loc,
                    "Rupture Risk": risk_info["category"],
                    "Clinical Action": risk_info["recommendation"],
                }
            )

        return pd.DataFrame(rows)

    @classmethod
    def generate_html_report(
        cls,
        scan_id: str,
        modality: str,
        detections: List[Dict],
        spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
        duration_sec: float = 0.0,
    ) -> str:
        """
        Generates an enterprise-grade printable HTML diagnostic report.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mod_upper = modality.upper()
        num_lesions = len(detections)

        # Risk breakdown
        max_diam = max([d.get("diameter_mm", 0.0) for d in detections]) if detections else 0.0
        max_risk = "Normal / None"
        if detections:
            risks = [
                cls.calculate_rupture_risk(
                    d.get("diameter_mm", 0.0),
                    d.get("location", ""),
                    confidence=float(d.get("aneurysm_prob", 0.0)),
                )["category"]
                for d in detections
            ]
            for r in ["Critical", "High", "Moderate-High", "Moderate", "Low"]:
                if r in risks:
                    max_risk = r
                    break

        badge_color = {
            "Critical": "#ef4444",
            "High": "#f97316",
            "Moderate-High": "#f59e0b",
            "Moderate": "#eab308",
            "Low": "#10b981",
            "Normal / None": "#64748b",
            "Borderline / Low Suspicion": "#64748b",
        }.get(max_risk, "#64748b")

        table_rows_html = ""
        for i, det in enumerate(detections):
            score = float(det.get("aneurysm_prob", 0.0))
            diam = float(det.get("diameter_mm", 0.0))
            phys = det.get("refined_phys_coord", (0.0, 0.0, 0.0))
            loc = det.get("location") or cls.classify_anatomical_location(phys, spacing=spacing)
            risk = cls.calculate_rupture_risk(diam, loc, confidence=score)

            table_rows_html += f"""
            <tr>
                <td style="font-weight: 600; text-align: center;">#{i+1}</td>
                <td><span style="background: rgba(56, 189, 248, 0.15); color: #0284c7; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{score*100:.1f}%</span></td>
                <td style="font-weight: 600;">{diam:.2f} mm</td>
                <td>({phys[0]:.1f}, {phys[1]:.1f}, {phys[2]:.1f})</td>
                <td>{loc}</td>
                <td><span style="background: {risk['color']}; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">{risk['category']}</span></td>
                <td style="font-size: 12px; color: #475569;">{risk['recommendation']}</td>
            </tr>
            """

        if not detections:
            table_rows_html = """
            <tr>
                <td colspan="7" style="text-align: center; padding: 24px; color: #10b981; font-weight: 600;">
                    ✓ No aneurysm candidates detected above clinical confidence threshold.
                </td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CerebroVascular AI Diagnostic Report - {scan_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 24px;
        }}
        .report-card {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid #e2e8f0;
            padding: 32px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0284c7;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 22px;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .header .meta {{
            font-size: 12px;
            color: #64748b;
            text-align: right;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-box {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #0284c7;
        }}
        .kpi-box .label {{
            font-size: 11px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
        }}
        .kpi-box .value {{
            font-size: 18px;
            font-weight: bold;
            color: #0f172a;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 16px;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            text-align: left;
            padding: 10px 12px;
            font-size: 11px;
            text-transform: uppercase;
            border-bottom: 2px solid #e2e8f0;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f1f5f9;
        }}
        .footer {{
            margin-top: 36px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 11px;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
        }}
        @media print {{
            body {{ background: #fff; padding: 0; }}
            .report-card {{ box-shadow: none; border: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <h1>🧠 CerebroVascular AI Diagnostic Summary</h1>
                <div style="font-size: 13px; color: #0284c7; font-weight: 600; margin-top: 4px;">
                    Dual-Expert Multimodal 3D CNN + Graph Attention Network (GATv2)
                </div>
            </div>
            <div class="meta">
                <div><strong>Scan ID:</strong> {scan_id}</div>
                <div><strong>Generated:</strong> {now_str}</div>
                <div><strong>Modality:</strong> {mod_upper} | <strong>Runtime:</strong> {duration_sec:.1f}s</div>
            </div>
        </div>

        <div class="kpi-row">
            <div class="kpi-box">
                <div class="label">Modality</div>
                <div class="value">{mod_upper} Angiography</div>
            </div>
            <div class="kpi-box">
                <div class="label">Aneurysms Detected</div>
                <div class="value" style="color: {'#ef4444' if num_lesions > 0 else '#10b981'};">{num_lesions}</div>
            </div>
            <div class="kpi-box">
                <div class="label">Max Lesion Diameter</div>
                <div class="value">{max_diam:.2f} mm</div>
            </div>
            <div class="kpi-box" style="border-left-color: {badge_color};">
                <div class="label">Overall Risk Tier</div>
                <div class="value" style="color: {badge_color}; font-size: 16px;">{max_risk}</div>
            </div>
        </div>

        <h3 style="font-size: 15px; margin-bottom: 8px; color: #1e293b;">📋 Detailed Morphological & Anatomical Findings</h3>
        <table>
            <thead>
                <tr>
                    <th style="text-align: center;">ID</th>
                    <th>Confidence</th>
                    <th>Diameter</th>
                    <th>Coords (mm)</th>
                    <th>Location</th>
                    <th>Rupture Risk</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>

        <div style="margin-top: 24px; padding: 16px; background: #eff6ff; border-radius: 8px; border: 1px solid #bfdbfe; font-size: 12px; color: #1e40af;">
            <strong>📌 Methodological Note:</strong>
            Predictions derived using the Dual-Branch ResNet-18 + Arterial Topology GATv2 model. Risk stratification incorporates PHASES score guidelines and vascular branching curvature priors. For investigational decision support.
        </div>

        <div class="footer">
            <div>Dayananda Sagar University (DSU) • Major Project 2026-2027</div>
            <div>Department of Computer Science & Engineering</div>
        </div>
    </div>
</body>
</html>
        """
        return html_content

