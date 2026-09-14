"""
Exploratory Data Analysis (EDA) & Lesion Size Profiler.
Measures physical diameter (mm), volume (mm^3), and anatomical distributions
for all aneurysms in the TopAneu 2026 dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def profile_dataset(
    dataset_root: str = "dataset(topAneu)",
    output_dir: str = "artifacts/eda",
    max_scans: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(dataset_root, "images")
    loc_masks_dir = os.path.join(dataset_root, "location_masks")
    loc_jsons_dir = os.path.join(dataset_root, "location_jsons")
    vessel_masks_dir = os.path.join(dataset_root, "vessel_masks")
    type_masks_dir = os.path.join(dataset_root, "type_masks")
    loc_map_path = os.path.join(dataset_root, "location_mapping.json")
    type_map_path = os.path.join(dataset_root, "type_mapping.json")

    # Load mappings
    with open(loc_map_path, "r") as f:
        loc_data = json.load(f)
        loc_mapping = loc_data.get("labels", loc_data)
        id_to_loc = {int(v): k for k, v in loc_mapping.items()}

    id_to_type = {}
    if os.path.exists(type_map_path):
        with open(type_map_path, "r") as f:
            t_data = json.load(f)
            t_map = t_data.get("labels", t_data)
            id_to_type = {int(v): k for k, v in t_map.items()}

    scan_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    if max_scans:
        scan_files = scan_files[:max_scans]

    print(f"Profiling {len(scan_files)} scans for aneurysm morphometry...")
    records = []

    for img_file in tqdm(scan_files, desc="Profiling scans"):
        base = img_file.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        parts = base.split("_")
        center, mod, pt_id = parts[1], parts[2], parts[3]

        mask_file = f"{base}.nii.gz"
        mask_path = os.path.join(loc_masks_dir, mask_file)
        if not os.path.exists(mask_path):
            continue

        try:
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata().astype(np.int32)
            spacing = np.array(mask_nii.header.get_zooms()[:3], dtype=np.float32)
        except Exception as e:
            print(f"Error loading {mask_file}: {e}")
            continue

        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            continue

        # Load type mask if available
        type_mask_path = os.path.join(type_masks_dir, mask_file)
        type_data = None
        if os.path.exists(type_mask_path):
            try:
                type_data = nib.load(type_mask_path).get_fdata().astype(np.int32)
            except:
                pass

        for lbl in unique_labels:
            voxels = np.argwhere(mask_data == lbl)
            n_voxels = len(voxels)
            if n_voxels == 0:
                continue

            phys_coords = voxels * spacing
            vol_mm3 = float(n_voxels * np.prod(spacing))
            equiv_diam = float(2.0 * ((3.0 * vol_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0))

            # Caliper maximum 3D distance
            if n_voxels > 1:
                if n_voxels > 2000:
                    sample_idx = np.random.choice(n_voxels, size=1000, replace=False)
                    max_diam = float(np.max(pdist(phys_coords[sample_idx])))
                else:
                    max_diam = float(np.max(pdist(phys_coords)))
            else:
                max_diam = float(np.max(spacing))

            # Centroid
            centroid_voxel = voxels.mean(axis=0).tolist()
            centroid_phys = (voxels.mean(axis=0) * spacing).tolist()

            # Category
            if max_diam < 3.0:
                size_cat = "Very Small (<3mm)"
            elif max_diam <= 5.0:
                size_cat = "Small (3-5mm)"
            elif max_diam <= 10.0:
                size_cat = "Medium (5-10mm)"
            else:
                size_cat = "Large (>10mm)"

            loc_name = id_to_loc.get(int(lbl), f"Unknown ({lbl})")

            # Morphology type
            morph_type = "saccular"
            if type_data is not None:
                type_vals = type_data[mask_data == lbl]
                if len(type_vals) > 0:
                    dominant_type = int(pd.Series(type_vals).mode()[0])
                    morph_type = id_to_type.get(dominant_type, "saccular")

            records.append({
                "scan_id": base,
                "center": center,
                "modality": mod,
                "label_id": int(lbl),
                "location_name": loc_name,
                "morphology": morph_type,
                "voxel_count": n_voxels,
                "volume_mm3": round(vol_mm3, 2),
                "max_diameter_mm": round(max_diam, 2),
                "equiv_diameter_mm": round(equiv_diam, 2),
                "size_category": size_cat,
                "centroid_voxel_x": round(centroid_voxel[0], 1),
                "centroid_voxel_y": round(centroid_voxel[1], 1),
                "centroid_voxel_z": round(centroid_voxel[2], 1),
                "spacing_x": round(float(spacing[0]), 3),
                "spacing_y": round(float(spacing[1]), 3),
                "spacing_z": round(float(spacing[2]), 3)
            })

    df = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, "aneurysm_annotations_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nAneurysm summary table saved to {csv_out}")
    print(f"Total aneurysms analyzed: {len(df)}")

    # Print Size Distribution
    print("\n" + "=" * 60)
    print("SIZE STRATIFICATION BREAKDOWN (RESEARCH CORE)")
    print("=" * 60)
    size_counts = df["size_category"].value_counts()
    for cat, count in size_counts.items():
        pct = (count / len(df)) * 100
        print(f"{cat:<22} : {count:>4} cases ({pct:>5.1f}%)")
    print("=" * 60)

    # Generate Publication-Ready Visualizations
    generate_eda_plots(df, output_dir)
    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: str):
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Plot 1: Lesion Diameter Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="max_diameter_mm",
        hue="size_category",
        bins=30,
        kde=True,
        ax=ax,
        palette={"Very Small (<3mm)": "#e74c3c", "Small (3-5mm)": "#e67e22", "Medium (5-10mm)": "#3498db", "Large (>10mm)": "#9b59b6"}
    )
    ax.axvline(3.0, color="red", linestyle="--", linewidth=1.5, label="Small Cutoff (3mm)")
    ax.axvline(5.0, color="orange", linestyle="--", linewidth=1.5, label="Medium Cutoff (5mm)")
    ax.set_title("Intracranial Aneurysm Maximum Diameter Distribution (TopAneu 2026)", fontweight="bold")
    ax.set_xlabel("Maximum 3D Diameter (mm)")
    ax.set_ylabel("Number of Aneurysms")
    plt.legend()
    plot1_path = os.path.join(output_dir, "size_distribution.png")
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot1_path}")

    # Plot 2: Size Distribution across Modalities
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="modality", y="max_diameter_mm", hue="center", ax=ax, palette="Set2")
    ax.set_title("Aneurysm Diameter across Modalities (CTA vs MRA) & Centers", fontweight="bold")
    ax.set_xlabel("Imaging Modality (ct = CTA, mr = TOF-MRA)")
    ax.set_ylabel("Maximum Diameter (mm)")
    ax.set_ylim(0, 20)
    plot2_path = os.path.join(output_dir, "modality_center_size_distribution.png")
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot2_path}")

    # Plot 3: Top 10 Anatomical Locations
    fig, ax = plt.subplots(figsize=(10, 6))
    top_locs = df["location_name"].value_counts().nlargest(10).index
    sns.countplot(
        data=df[df["location_name"].isin(top_locs)],
        y="location_name",
        order=top_locs,
        hue="size_category",
        ax=ax,
        palette="rocket"
    )
    ax.set_title("Top 10 Anatomical Aneurysm Locations by Size Category", fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Anatomical Vessel Branch")
    plot3_path = os.path.join(output_dir, "top_anatomical_locations.png")
    fig.savefig(plot3_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot3_path}")


if __name__ == "__main__":
    profile_dataset()
