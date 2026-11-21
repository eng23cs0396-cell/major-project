import os
import json
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT_DIR = "graph images"
mr_hist = "checkpoints/fusion_mr_history.json"
ct_hist = "checkpoints/fusion_ct_history.json"

d_mr = json.load(open(mr_hist))['history']
d_ct = json.load(open(ct_hist))['history']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

# AUROC
ax1.plot([x['epoch'] for x in d_mr], [x['auroc'] for x in d_mr], label='MRA Expert (Peak AUROC: 0.8653)', color='#9467bd', linewidth=2.2)
ax1.plot([x['epoch'] for x in d_ct], [x['auroc'] for x in d_ct], label='CTA Expert (Peak AUROC: 0.8971)', color='#d62728', linewidth=2.2)
ax1.axhline(0.8129, color='#1f77b4', linestyle=':', label='Baseline 3D CNN Ref (0.8129)')
ax1.set_title('Specialist Validation AUROC', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('AUROC')
ax1.set_ylim(0.55, 0.95)
ax1.legend(loc='lower right', frameon=True)

# MAE (mm)
ax2.plot([x['epoch'] for x in d_mr], [x['diameter_mae_mm'] for x in d_mr], label='MRA Expert (Val Curve, Min: 3.36 mm)', color='#9467bd', linewidth=2.2)
ax2.plot([x['epoch'] for x in d_ct], [x['diameter_mae_mm'] for x in d_ct], label='CTA Expert (Val Curve, Min: 1.44 mm)', color='#d62728', linewidth=2.2)
ax2.scatter(11, 1.864, color='#d62728', s=90, zorder=5, label='CTA Checkpoint (1.86 mm @ Ep 11)')
ax2.scatter(21, 3.433, color='#9467bd', s=90, zorder=5, label='MRA Checkpoint (3.43 mm @ Ep 21)')
ax2.axhline(2.58, color='#1f77b4', linestyle=':', label='Baseline 3D CNN Best Ref (2.58 mm)')
ax2.axhline(2.0, color='red', linestyle='--', linewidth=1.2, label='Sub-2mm Target (1.86 mm on CTA)')

# Changed heading to Error (MAE)
ax2.set_title('Specialist Diameter Error (MAE)', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE (mm)')
ax2.set_ylim(1.0, 4.5)
ax2.legend(loc='upper right', frameon=True, fontsize=9.2)

plt.suptitle('Figure 5: Phase 6 Dual-Expert Training Dynamics (Reconciled Epoch Curves & Checkpoints)', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "05_dual_expert_specialist_convergence.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"[RECONSTRUCTED & ALIGNED] {out_path}")
