"""
Patient-Stratified Dataset Splitter for TopAneu 2026.
Ensures zero data leakage across patient longitudinal scans and maintains
balanced proportions of modalities, centers, and healthy/diseased cases.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from collections import defaultdict


def extract_patient_info(filename: str, jsons_dir: str):
    """
    Parse filename schema: topaneu_{centerID}_{modality}_{patientID}[_scanNum]_0000.nii.gz
    """
    base = filename.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    parts = base.split("_")
    center = parts[1]      # center1, center2, center4, center5
    modality = parts[2]    # mr or ct
    patient_id = parts[3]  # unique patient identifier
    
    # Center 4 can have longitudinal scans (e.g. topaneu_center4_ct_008_1)
    unique_patient_key = f"{center}_{modality}_{patient_id}"
    
    # Check aneurysm presence
    json_name = f"{base}.json"
    json_path = os.path.join(jsons_dir, json_name)
    num_aneurysms = 0
    locations = []
    if os.path.exists(json_path):
        with open(json_path, "r") as fp:
            data = json.load(fp)
            locations = data.get("locations", [])
            num_aneurysms = len(locations)

    return {
        "filename": filename,
        "scan_id": base,
        "center": center,
        "modality": modality,
        "patient_id": patient_id,
        "patient_key": unique_patient_key,
        "num_aneurysms": num_aneurysms,
        "has_aneurysm": int(num_aneurysms > 0),
        "locations": ";".join(str(l) for l in locations)
    }


def generate_splits(
    dataset_root: str = "dataset(topAneu)",
    output_dir: str = "configs",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    np.random.seed(seed)
    images_dir = os.path.join(dataset_root, "images")
    jsons_dir = os.path.join(dataset_root, "location_jsons")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    print(f"Total scan files located: {len(files)}")

    records = [extract_patient_info(f, jsons_dir) for f in files]
    df = pd.DataFrame(records)

    # Save complete dataset manifest
    manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"Dataset manifest saved to {manifest_path}")

    # Group by unique patient key to prevent leakage of longitudinal scans
    patient_df = df.groupby("patient_key").agg({
        "center": "first",
        "modality": "first",
        "has_aneurysm": "max",
        "num_aneurysms": "sum",
        "scan_id": "count"
    }).reset_index().rename(columns={"scan_id": "num_scans"})

    print(f"Total unique patients: {len(patient_df)}")

    # Stratified split based on stratum = (center + modality + has_aneurysm)
    patient_df["stratum"] = (
        patient_df["center"] + "_" + 
        patient_df["modality"] + "_" + 
        patient_df["has_aneurysm"].astype(str)
    )

    train_patients = []
    val_patients = []
    test_patients = []

    for stratum, group in patient_df.groupby("stratum"):
        shuffled = group["patient_key"].values
        np.random.shuffle(shuffled)
        n = len(shuffled)
        
        n_val = max(1, int(round(n * val_ratio))) if n >= 4 else (1 if n >= 2 else 0)
        n_test = max(1, int(round(n * test_ratio))) if n >= 4 else (1 if n >= 3 else 0)
        n_train = n - n_val - n_test
        
        if n_train <= 0:
            n_train = max(1, n - 1)
            n_val = n - n_train
            n_test = 0

        train_pts = shuffled[:n_train]
        val_pts = shuffled[n_train:n_train + n_val]
        test_pts = shuffled[n_train + n_val:]

        train_patients.extend(train_pts)
        val_patients.extend(val_pts)
        test_patients.extend(test_pts)

    # Map back to all scans
    train_df = df[df["patient_key"].isin(train_patients)].copy()
    val_df = df[df["patient_key"].isin(val_patients)].copy()
    test_df = df[df["patient_key"].isin(test_patients)].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Verify zero patient overlap
    overlap = set(train_df["patient_key"]).intersection(set(val_df["patient_key"])).union(
        set(train_df["patient_key"]).intersection(set(test_df["patient_key"]))
    ).union(
        set(val_df["patient_key"]).intersection(set(test_df["patient_key"]))
    )
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping patients: {overlap}"

    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("PATIENT-STRATIFIED SPLIT SUMMARY")
    print("=" * 60)
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_mra = len(split_df[split_df["modality"] == "mr"])
        n_cta = len(split_df[split_df["modality"] == "ct"])
        n_pos = len(split_df[split_df["has_aneurysm"] == 1])
        n_neg = len(split_df[split_df["has_aneurysm"] == 0])
        n_aneu = split_df["num_aneurysms"].sum()
        print(f"{name:<6} | Total: {len(split_df):^4} | MRA: {n_mra:^3} | CTA: {n_cta:^3} | Pos: {n_pos:^3} | Neg: {n_neg:^3} | Aneurysms: {n_aneu:^3}")
    print("=" * 60)

    # Also generate Center-4 Holdout Split for Cross-Hospital Benchmark
    c4_test_df = df[df["center"] == "center4"].copy()
    c4_train_df = df[df["center"] != "center4"].copy()
    c4_test_df["split"] = "test_holdout_center4"
    c4_train_df["split"] = "train_holdout_center4"

    c4_holdout_path = os.path.join(output_dir, "center4_holdout_split.csv")
    pd.concat([c4_train_df, c4_test_df]).to_csv(c4_holdout_path, index=False)
    print(f"Cross-hospital Center-4 holdout split saved to {c4_holdout_path}\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    generate_splits()
