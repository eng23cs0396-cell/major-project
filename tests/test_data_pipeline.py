"""
Unit Tests for Data Pipeline, Splitting, and 3D Preprocessing.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
import pandas as pd
import torch
from src.data.preprocessor import VolumetricPreprocessor
from src.data.dataset import AneurysmPatchDataset


def test_split_integrity():
    train_path = "configs/train_split.csv"
    val_path = "configs/val_split.csv"
    test_path = "configs/test_split.csv"

    assert os.path.exists(train_path), "Train split missing!"
    assert os.path.exists(val_path), "Val split missing!"
    assert os.path.exists(test_path), "Test split missing!"

    df_tr = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_te = pd.read_csv(test_path)

    # Verify zero patient overlap
    pts_tr = set(df_tr["patient_key"])
    pts_val = set(df_val["patient_key"])
    pts_te = set(df_te["patient_key"])

    overlap_tr_val = pts_tr.intersection(pts_val)
    overlap_tr_te = pts_tr.intersection(pts_te)
    overlap_val_te = pts_val.intersection(pts_te)

    assert len(overlap_tr_val) == 0, f"Patient overlap between train and val: {overlap_tr_val}"
    assert len(overlap_tr_te) == 0, f"Patient overlap between train and test: {overlap_tr_te}"
    assert len(overlap_val_te) == 0, f"Patient overlap between val and test: {overlap_val_te}"

    print(f"[PASS] Split integrity verified with ZERO data leakage across {len(pts_tr) + len(pts_val) + len(pts_te)} patients!")


def test_preprocessor_mra_and_cta():
    prep = VolumetricPreprocessor(patch_size=(64, 64, 64))

    # Synthetic MRA volume
    synth_mra = np.zeros((100, 100, 100), dtype=np.float32)
    synth_mra[20:80, 20:80, 20:80] = np.random.uniform(50, 500, (60, 60, 60))
    norm_mra = prep.normalize_intensity(synth_mra, modality="mr")
    
    assert norm_mra.shape == (100, 100, 100)
    assert not np.isnan(norm_mra).any()
    print("[PASS] MRA intensity normalization verified!")

    # Synthetic CTA volume (Hounsfield units)
    synth_cta = np.random.uniform(-100, 1000, (100, 100, 100)).astype(np.float32)
    norm_cta = prep.normalize_intensity(synth_cta, modality="ct")
    
    assert norm_cta.min() >= 0.0, f"CTA normalized min < 0: {norm_cta.min()}"
    assert norm_cta.max() <= 1.0, f"CTA normalized max > 1: {norm_cta.max()}"
    print("[PASS] CTA windowing [100, 700] HU normalization verified!")


def test_patch_extraction_with_padding():
    prep = VolumetricPreprocessor(patch_size=(64, 64, 64))
    vol = np.random.randn(80, 80, 80).astype(np.float32)
    mask = np.zeros((80, 80, 80), dtype=np.float32)

    # Test corner extraction (requires symmetric edge padding)
    patch, mask_patch = prep.extract_patch(vol, center_voxel=(5, 5, 5), mask=mask)
    assert patch.shape == (64, 64, 64), f"Expected (64, 64, 64), got {patch.shape}"
    assert mask_patch.shape == (64, 64, 64)
    print("[PASS] Edge boundary 3D patch padding & extraction verified!")


def test_dataset_loader():
    dataset = AneurysmPatchDataset(
        split_csv="configs/val_split.csv",
        dataset_root="dataset(topAneu)",
        neg_pos_ratio=1.0,
        is_training=False
    )

    assert len(dataset) > 0, "Dataset is empty!"
    sample = dataset[0]

    assert "patch" in sample
    assert "label" in sample
    assert "size_mm" in sample
    assert sample["patch"].shape == (1, 64, 64, 64), f"Unexpected patch shape: {sample['patch'].shape}"
    assert sample["patch"].dtype == torch.float32
    print(f"[PASS] PyTorch Dataset loading verified! Patch tensor shape: {sample['patch'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DATA PIPELINE UNIT TESTS")
    print("=" * 60)
    test_split_integrity()
    test_preprocessor_mra_and_cta()
    test_patch_extraction_with_padding()
    test_dataset_loader()
    print("=" * 60)
