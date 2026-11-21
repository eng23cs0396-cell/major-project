import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

OUT_DIR = "graph images"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Profiling: Size Distribution (Phase 2)
# -------------------------------------------------------------
eda_csv = "artifacts/eda/aneurysm_annotations_summary.csv"
if os.path.exists(eda_csv):
    df_eda = pd.read_csv(eda_csv)
    diameters = df_eda['max_diameter_mm'].dropna()
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(diameters, bins=35, kde=True, color='#1f77b4', edgecolor='black', alpha=0.6, ax=ax)
    ax.axvline(3.0, color='#d62728', linestyle='--', linewidth=2, label='Very Small Cutoff (3 mm)')
    ax.axvline(5.0, color='#ff7f0e', linestyle='--', linewidth=2, label='Small Cutoff (5 mm)')
    ax.set_title('Figure 1: Distribution of Intracranial Aneurysm Maximum Diameter (TopAneu 2026)', pad=12, fontweight='bold')
    ax.set_xlabel('Maximum 3D Diameter (mm)')
    ax.set_ylabel('Number of Aneurysms')
    ax.set_xlim(0, 35)
    ax.legend(frameon=True, loc='upper right')
    
    # Annotate clinical share
    n_small = (diameters <= 5.0).sum()
    pct_small = (n_small / len(diameters)) * 100
    ax.text(0.55, 0.65, f'Aneurysms ≤ 5 mm: {n_small}/{len(diameters)} ({pct_small:.1f}%)\nMedian: {diameters.median():.2f} mm | Mean: {diameters.mean():.2f} mm',
            transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#ccc', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_dataset_aneurysm_size_distribution.png"))
    plt.close()
    print("[DONE] 01_dataset_aneurysm_size_distribution.png")

# -------------------------------------------------------------
# 2. Dataset Profiling: Modality & Multi-Center Distribution (Phase 2)
# -------------------------------------------------------------
if os.path.exists(eda_csv):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.boxplot(data=df_eda, x='modality', y='max_diameter_mm', hue='center', palette='Set2', ax=ax, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    ax.set_title('Figure 2: Aneurysm Diameter across Imaging Modalities and Hospital Centers', pad=12, fontweight='bold')
    ax.set_xlabel('Imaging Modality (ct = CTA, mr = TOF-MRA)')
    ax.set_ylabel('Maximum Diameter (mm)')
    ax.set_ylim(0, 25)
    ax.legend(title='Center Code', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_dataset_modality_center_boxplots.png"))
    plt.close()
    print("[DONE] 02_dataset_modality_center_boxplots.png")

# -------------------------------------------------------------
# 3. Dataset Profiling: Top Anatomical Sites (Phase 2)
# -------------------------------------------------------------
if os.path.exists(eda_csv) and 'location' in df_eda.columns:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top_locs = df_eda['location'].value_counts().head(10).index
    df_top = df_eda[df_eda['location'].isin(top_locs)]
    
    # Categorize sizes
    def size_cat(d):
        if d < 3.0: return '< 3 mm (Very Small)'
        elif d <= 5.0: return '3 - 5 mm (Small)'
        elif d <= 10.0: return '5 - 10 mm (Medium)'
        else: return '> 10 mm (Large)'
    df_top['size_group'] = df_top['max_diameter_mm'].apply(size_cat)
    
    sns.countplot(data=df_top, y='location', hue='size_group', order=top_locs,
                  palette='magma_r', ax=ax)
    ax.set_title('Figure 3: Top 10 Anatomical Aneurysm Locations Stratified by Size Category', pad=12, fontweight='bold')
    ax.set_xlabel('Lesion Count')
    ax.set_ylabel('Anatomical Arterial Site')
    ax.legend(title='Size Category', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_dataset_top_anatomical_locations.png"))
    plt.close()
    print("[DONE] 03_dataset_top_anatomical_locations.png")

# -------------------------------------------------------------
# 4. Training Dynamics: Baseline vs. First Fusion Model (Phase 3 vs Phase 5)
# -------------------------------------------------------------
base_hist = "checkpoints/baseline_train_history.json"
fuse_hist = "checkpoints/fusion_train_history.json"
if os.path.exists(base_hist) and os.path.exists(fuse_hist):
    d_base = json.load(open(base_hist))['history']
    d_fuse = json.load(open(fuse_hist))['history']
    
    ep_base = [x['epoch'] for x in d_base]
    ep_fuse = [x['epoch'] for x in d_fuse]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    
    # AUROC
    ax1.plot(ep_base, [x['auroc'] for x in d_base], label='Phase 3 Baseline (3D ResNet-18)', color='#1f77b4', linewidth=2)
    ax1.plot(ep_fuse, [x['auroc'] for x in d_fuse], label='Phase 5 Dual-Branch Fusion (CNN+GATv2)', color='#2ca02c', linewidth=2)
    ax1.set_title('Validation AUROC Progression', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUROC')
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(0.45, 0.90)
    
    # Loss
    ax2.plot(ep_base, [x['val_loss'] for x in d_base], label='Baseline Val Loss', color='#1f77b4', linewidth=2)
    ax2.plot(ep_fuse, [x['val_loss'] for x in d_fuse], label='Fusion Val Loss', color='#2ca02c', linewidth=2)
    ax2.set_title('Validation Total Multi-Task Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right', frameon=True)
    
    # Diameter MAE
    ax3.plot(ep_base, [x['diameter_mae_mm'] for x in d_base], label='Baseline MAE (mm)', color='#1f77b4', linewidth=2)
    ax3.plot(ep_fuse, [x['diameter_mae_mm'] for x in d_fuse], label='Fusion MAE (mm)', color='#2ca02c', linewidth=2)
    ax3.set_title('Validation Diameter Regression MAE', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE (mm)')
    ax3.legend(loc='upper right', frameon=True)
    
    plt.suptitle('Figure 4: Patch-Level Convergence: Pure Visual Baseline vs. Topology-Aware Fusion', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_patch_level_learning_curves.png"), bbox_inches='tight')
    plt.close()
    print("[DONE] 04_patch_level_learning_curves.png")

# -------------------------------------------------------------
# 5. Dual-Expert Convergence: MRA Specialist vs CTA Specialist (Phase 6)
# -------------------------------------------------------------
mr_hist = "checkpoints/fusion_mr_history.json"
ct_hist = "checkpoints/fusion_ct_history.json"
if os.path.exists(mr_hist) and os.path.exists(ct_hist):
    d_mr = json.load(open(mr_hist))['history']
    d_ct = json.load(open(ct_hist))['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # AUROC
    ax1.plot([x['epoch'] for x in d_mr], [x['auroc'] for x in d_mr], label='MRA Expert (`best_fusion_mra.pt`)', color='#9467bd', linewidth=2.2)
    ax1.plot([x['epoch'] for x in d_ct], [x['auroc'] for x in d_ct], label='CTA Expert (`best_fusion_cta.pt`)', color='#d62728', linewidth=2.2)
    ax1.axhline(0.8129, color='#1f77b4', linestyle=':', label='Baseline 3D CNN Ref (0.8129)')
    ax1.set_title('Specialist Validation AUROC', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUROC')
    ax1.set_ylim(0.55, 0.95)
    ax1.legend(loc='lower right', frameon=True)
    
    # MAE
    ax2.plot([x['epoch'] for x in d_mr], [x['diameter_mae_mm'] for x in d_mr], label='MRA Expert (Best: 2.08 mm)', color='#9467bd', linewidth=2.2)
    ax2.plot([x['epoch'] for x in d_ct], [x['diameter_mae_mm'] for x in d_ct], label='CTA Expert (Best: 1.86 mm)', color='#d62728', linewidth=2.2)
    ax2.axhline(2.58, color='#1f77b4', linestyle=':', label='Baseline 3D CNN Ref (2.58 mm)')
    ax2.set_title('Specialist Diameter Error (MAE mm)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (mm)')
    ax2.legend(loc='upper right', frameon=True)
    
    plt.suptitle('Figure 5: Phase 6 Dual-Expert Training Dynamics (Elimination of Modality Conflict)', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_dual_expert_specialist_convergence.png"), bbox_inches='tight')
    plt.close()
    print("[DONE] 05_dual_expert_specialist_convergence.png")

# -------------------------------------------------------------
# 6. Patch-Level Model Evolution: AUROC, Sensitivity, Specificity, MAE (Table 8.2)
# -------------------------------------------------------------
models = ['3D ResNet-18\nBaseline', 'Fusion\n(First Unified)', 'Fusion\n(Option C)', 'MRA Expert\n(Specialist)', 'CTA Expert\n(Specialist)']
aurocs = [0.8129, 0.8728, 0.8410, 0.8653, 0.8971]
sens = [79.2, 89.6, 82.4, 87.1, 94.1]
specs = [72.9, 79.2, 78.9, 80.0, 87.5]
maes = [2.89, 2.74, 2.14, 2.08, 1.86]

x = np.arange(len(models))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Classification metrics
ax1.bar(x - width, aurocs, width, label='AUROC', color='#1f77b4', edgecolor='black', alpha=0.85)
ax1.bar(x, [s/100 for s in sens], width, label='Sensitivity @ OP', color='#2ca02c', edgecolor='black', alpha=0.85)
ax1.bar(x + width, [sp/100 for sp in specs], width, label='Specificity @ OP', color='#ff7f0e', edgecolor='black', alpha=0.85)
ax1.set_ylabel('Score (0.0 to 1.0)')
ax1.set_title('Classification Performance across Architecture Iterations', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0.5, 1.0)
ax1.legend(loc='lower left', frameon=True)

# Regression MAE
bars = ax2.bar(models, maes, color=['#7f7f7f', '#aec7e8', '#98df8a', '#9467bd', '#d62728'], edgecolor='black', width=0.5, alpha=0.85)
ax2.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Sub-2mm Clinical Precision Mark')
ax2.set_ylabel('Diameter Error MAE (mm)')
ax2.set_title('Physical Diameter Regression Precision', fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} mm", ha='center', va='bottom', fontweight='bold')
ax2.set_ylim(0, 3.5)
ax2.legend(loc='upper right', frameon=True)

plt.suptitle('Figure 6: Comprehensive Patch-Level Validation across Model Generations (Table 8.2)', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_patch_level_benchmark_comparison.png"), bbox_inches='tight')
plt.close()
print("[DONE] 06_patch_level_benchmark_comparison.png")

# -------------------------------------------------------------
# 7. Whole-Scan FROC Curve: Validation Scans (Table 8.3)
# -------------------------------------------------------------
fp_rates = [0.5, 1.0, 2.0, 4.0]
base_froc = [6.0, 14.0, 20.0, 20.0]
fusion1_froc = [10.0, 22.9, 46.7, 60.0]
optionc_froc = [8.6, 20.0, 60.0, 60.0]

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
ax.plot(fp_rates, base_froc, 'o--', color='#1f77b4', linewidth=2.2, markersize=8, label='Phase 3 Baseline (Mean: 16.0%)')
ax.plot(fp_rates, fusion1_froc, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='Phase 5 First Fusion (Mean: 39.9%)')
ax.plot(fp_rates, optionc_froc, '^-.', color='#ff7f0e', linewidth=2.2, markersize=8, label='Phase 6 Option C Unified (Mean: 41.7%)')

ax.set_title('Figure 7: Whole-Scan Free-Response ROC (FROC) on Validation Scans', pad=12, fontweight='bold')
ax.set_xlabel('False Positives per Scan (FPs/Scan)')
ax.set_ylabel('Detection Sensitivity (%)')
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 75)
ax.axvline(2.0, color='gray', linestyle=':', alpha=0.7, label='Clinical Benchmark Operating Point (2.0 FP/scan)')
ax.legend(frameon=True, loc='lower right')

for x_pt, y_pt in zip(fp_rates, optionc_froc):
    ax.annotate(f'{y_pt:.1f}%', (x_pt, y_pt), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold', color='#ff7f0e')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_whole_scan_froc_validation_scans.png"))
plt.close()
print("[DONE] 07_whole_scan_froc_validation_scans.png")

# -------------------------------------------------------------
# 8. Center-4 Cross-Modality CTA Stress Test (Table 8.4)
# -------------------------------------------------------------
cta_fp = [0.5, 1.0, 2.0, 4.0]
cta_base = [0.0, 0.0, 0.0, 0.0]
cta_fuse1 = [6.7, 20.0, 30.0, 30.0]
cta_optc = [5.0, 11.7, 20.0, 26.0]
cta_expert = [9.2, 21.5, 44.6, 72.9]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
ax.plot(cta_fp, cta_base, 'x--', color='#7f7f7f', linewidth=2, markersize=8, label='3D ResNet-18 Baseline (Mean: 0.0% - Domain Collapse)')
ax.plot(cta_fp, cta_fuse1, 'o--', color='#2ca02c', linewidth=2, markersize=7, label='First Fusion Model (Mean: 21.7%)')
ax.plot(cta_fp, cta_optc, 'v--', color='#ff7f0e', linewidth=2, markersize=7, label='Option C Unified Model (Mean: 15.7%)')
ax.plot(cta_fp, cta_expert, 's-', color='#d62728', linewidth=3, markersize=9, label='CTA Specialist Expert (Mean: 37.1% | 72.9% @ 4 FP)')

ax.set_title('Figure 8: Domain Adaptation Stress Test: Center-4 CTA Cohort FROC Benchmark', pad=12, fontweight='bold')
ax.set_xlabel('False Positives per Scan (FPs/Scan)')
ax.set_ylabel('Lesion Sensitivity (%)')
ax.set_xlim(0, 4.5)
ax.set_ylim(-2, 85)
ax.legend(frameon=True, loc='upper left')

for x_pt, y_pt in zip(cta_fp, cta_expert):
    ax.annotate(f'{y_pt:.1f}%', (x_pt, y_pt), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold', color='#d62728')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_center4_cta_generalization_froc.png"))
plt.close()
print("[DONE] 08_center4_cta_generalization_froc.png")

# -------------------------------------------------------------
# 9. Size-Stratified Breakdown: The Small Aneurysm Thesis Proof (Table 8.5 & results.md)
# -------------------------------------------------------------
categories = ['Very Small\n(< 3 mm)', 'Small\n(3 - 5 mm)', 'Overall Small\n(≤ 5 mm)', 'Medium & Large\n(> 5 mm)', 'Overall Cohort']
sens_base = [0.0, 0.0, 0.0, 0.0, 0.0]
sens_optc = [0.0, 16.7, 12.0, 30.0, 26.0]
sens_expert = [14.3, 91.7, 85.9, 60.0, 72.9]

x = np.arange(len(categories))
width = 0.28

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
b1 = ax.bar(x - width, sens_base, width, label='3D CNN Baseline', color='#7f7f7f', edgecolor='black', alpha=0.7)
b2 = ax.bar(x, sens_optc, width, label='Unified Model (Option C)', color='#ff7f0e', edgecolor='black', alpha=0.85)
b3 = ax.bar(x + width, sens_expert, width, label='Dual-Expert CTA Specialist', color='#2ca02c', edgecolor='black', alpha=0.9)

ax.set_ylabel('Sensitivity @ 4.0 FP/Scan (%)')
ax.set_title('Figure 9: Lesion-Size Stratified Detection Sensitivity on Center-4 CTA (Core Thesis)', pad=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 105)
ax.legend(frameon=True, loc='upper left')

for bar in b2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d95f02')
for bar in b3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1b9e77')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_size_stratified_cta_sensitivity_breakdown.png"))
plt.close()
print("[DONE] 09_size_stratified_cta_sensitivity_breakdown.png")

# -------------------------------------------------------------
# 10. Vessel Mask Ablation Study: Quantitative Comparison (Table 8.6)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

labels = ['Raw Intensity Fallback\n(Without Vessel Mask)', 'Arterial Prior\n(With Vessel Mask)']
colors = ['#d62728', '#2ca02c']

# Metric 1: Search Space & Candidates
search_voxels = [1096366 / 1000, 284503 / 1000] # in thousands
b = ax1.bar(labels, search_voxels, color=colors, edgecolor='black', width=0.55, alpha=0.85)
ax1.set_ylabel('Search Space (Thousand Voxels)')
ax1.set_title('Search Space Volume', fontweight='bold')
ax1.text(0, search_voxels[0]+20, f"{int(search_voxels[0])}k voxels", ha='center', fontweight='bold')
ax1.text(1, search_voxels[1]+20, f"{int(search_voxels[1])}k voxels\n(-74.1%)", ha='center', fontweight='bold')
ax1.set_ylim(0, 1300)

# Metric 2: False Detections on Confirmed Healthy Scan
detections = [43, 12]
b = ax2.bar(labels, detections, color=colors, edgecolor='black', width=0.55, alpha=0.85)
ax2.set_ylabel('Reported Candidates (False Alarms)')
ax2.set_title('False Alarms on Healthy Scan', fontweight='bold')
ax2.text(0, detections[0]+1, "43 FPs\n(Scalp/Eyes/Bones)", ha='center', fontweight='bold')
ax2.text(1, detections[1]+1, "12 FPs\n(-72.1% Drop)", ha='center', fontweight='bold')
ax2.set_ylim(0, 52)

# Metric 3: CPU Inference Latency
latencies = [877.3 / 60.0, 133.8 / 60.0] # in minutes
b = ax3.bar(labels, latencies, color=colors, edgecolor='black', width=0.55, alpha=0.85)
ax3.set_ylabel('CPU End-to-End Latency (Minutes)')
ax3.set_title('Inference Runtime', fontweight='bold')
ax3.text(0, latencies[0]+0.3, f"{latencies[0]:.1f} min", ha='center', fontweight='bold')
ax3.text(1, latencies[1]+0.3, f"{latencies[1]:.1f} min\n(6.5x Faster)", ha='center', fontweight='bold')
ax3.set_ylim(0, 17)

plt.suptitle('Figure 10: Quantitative Ablation of Arterial Vessel Mask on Healthy Control Scan (Table 8.6)', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_vessel_mask_impact_ablation.png"), bbox_inches='tight')
plt.close()
print("[DONE] 10_vessel_mask_impact_ablation.png")

print("\nAll 10 rigorous, publication-grade figures successfully generated in 'graph images/' directory!")
