import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

OUT_DIR = "graph images"
os.makedirs(OUT_DIR, exist_ok=True)

# Models and evaluated metrics
models = ['3D ResNet-18\nBaseline', 'Fusion\n(First Unified)', 'Fusion\n(Option C)', 'MRA Expert\n(Specialist)', 'CTA Expert\n(Specialist)']
aurocs = [0.8129, 0.8728, 0.8410, 0.8653, 0.8971]
sens = [79.2, 89.6, 82.4, 87.1, 94.1]
specs = [72.9, 79.2, 78.9, 80.0, 87.5]

# Verified MAE at selected optimal checkpoint (3.43 mm for MRA Expert, 1.86 mm for CTA Expert)
# Also note that across the training trajectory, MRA reached min MAE of 3.36 mm
maes_checkpoint = [2.89, 2.74, 2.14, 3.43, 1.86]

x = np.arange(len(models))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)

# Left Subplot: Classification Performance
ax1.bar(x - width, aurocs, width, label='AUROC', color='#1f77b4', edgecolor='black', alpha=0.85)
ax1.bar(x, [s/100 for s in sens], width, label="Sensitivity @ Youden's OP", color='#2ca02c', edgecolor='black', alpha=0.85)
ax1.bar(x + width, [sp/100 for sp in specs], width, label="Specificity @ Youden's OP", color='#ff7f0e', edgecolor='black', alpha=0.85)
ax1.set_ylabel('Score (0.0 to 1.0)')
ax1.set_title("Classification at Youden's Optimal Operating Point (OP)", fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0.5, 1.0)
ax1.legend(loc='lower left', frameon=True)

# Right Subplot: Physical Diameter Regression Error (MAE)
# Corrected Heading as requested: 'Physical Diameter Regression Error (MAE)'
bars = ax2.bar(models, maes_checkpoint, color=['#7f7f7f', '#aec7e8', '#98df8a', '#9467bd', '#d62728'], edgecolor='black', width=0.5, alpha=0.85)
ax2.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Sub-2mm Target (CTA Expert: 1.86 mm)')
ax2.set_ylabel('Diameter Regression MAE (mm)')
ax2.set_title('Physical Diameter Regression Error (MAE)', fontweight='bold')

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.06, f"{yval:.2f} mm", ha='center', va='bottom', fontweight='bold')

ax2.set_ylim(0, 4.2)
ax2.legend(loc='upper right', frameon=True)

plt.suptitle("Figure 6: Patch-Level Validation Metrics across Model Generations (OP = Youden's J-statistic)", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "06_patch_level_benchmark_comparison.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"[RECONSTRUCTED & CORRECTED] {out_path}")
