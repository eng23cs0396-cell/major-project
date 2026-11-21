import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT_DIR = "graph images"
eda_csv = "artifacts/eda/aneurysm_annotations_summary.csv"
df_eda = pd.read_csv(eda_csv)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
top_locs = df_eda['location_name'].value_counts().head(10).index
df_top = df_eda[df_eda['location_name'].isin(top_locs)].copy()

order_cats = ['Very Small (<3mm)', 'Small (3-5mm)', 'Medium (5-10mm)', 'Large (>10mm)']
palette = {
    'Very Small (<3mm)': '#d95f02',
    'Small (3-5mm)': '#e7298a',
    'Medium (5-10mm)': '#7570b3',
    'Large (>10mm)': '#1b9e77'
}

sns.countplot(
    data=df_top,
    y='location_name',
    hue='size_category',
    hue_order=order_cats,
    order=top_locs,
    palette=palette,
    ax=ax
)

ax.set_title('Figure 3: Top 10 Anatomical Aneurysm Sites Stratified by Size Category', pad=12, fontweight='bold', fontsize=13)
ax.set_xlabel('Number of Lesions', fontsize=11, fontweight='bold')
ax.set_ylabel('Anatomical Arterial Site', fontsize=11, fontweight='bold')
ax.legend(title='Size Category', frameon=True, loc='lower right')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "03_dataset_top_anatomical_locations.png")
plt.savefig(out_path)
plt.close()
print(f"[DONE] Saved {out_path}")
