"""
Unit tests for the Phase 3 3D ResNet baseline: forward, loss, and backward.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch

from src.models.losses import MultiTaskDetectionLoss
from src.models.resnet3d import build_baseline_model, count_parameters


BATCH = 4
SHAPE = (BATCH, 1, 64, 64, 64)


def _dummy_batch(device: torch.device):
    patches = torch.randn(*SHAPE, device=device)
    labels = torch.tensor([1, 0, 1, 0], device=device)
    offsets = torch.tensor(
        [[1.0, -0.5, 0.25], [0.0, 0.0, 0.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        device=device,
    )
    sizes = torch.tensor([4.2, 0.0, 7.1, 0.0], device=device)
    return patches, labels, offsets, sizes


def _assert_forward_backward(device: torch.device) -> None:
    model = build_baseline_model(backbone="resnet18_3d", embedding_dim=512, dropout=0.0).to(device)
    criterion = MultiTaskDetectionLoss()
    patches, labels, offsets, sizes = _dummy_batch(device)

    model.train()
    outputs = model(patches)
    assert outputs["embedding"].shape == (BATCH, 512), outputs["embedding"].shape
    assert outputs["aneurysm_logit"].shape == (BATCH,)
    assert outputs["aneurysm_prob"].shape == (BATCH,)
    assert outputs["offset"].shape == (BATCH, 3)
    assert outputs["diameter_mm"].shape == (BATCH,)
    assert torch.all(outputs["aneurysm_prob"] >= 0) and torch.all(outputs["aneurysm_prob"] <= 1)
    assert torch.all(outputs["diameter_mm"] >= 0)

    losses = criterion(outputs, labels, offsets, sizes)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed."
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradient detected."
    print(f"[PASS] Forward/backward on {device} | loss={float(losses['loss'].detach()):.4f}")


def test_parameter_count():
    n = count_parameters()
    assert n > 1_000_000, f"Model too small: {n} params"
    print(f"[PASS] ResNet-18 3D trainable parameters: {n:,}")


def test_forward_backward_cpu():
    _assert_forward_backward(torch.device("cpu"))


def test_forward_backward_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    _assert_forward_backward(torch.device("cuda"))


def test_loss_positive_only_regression():
    criterion = MultiTaskDetectionLoss(lambda_offset=1.0, lambda_size=1.0)
    outputs = {
        "aneurysm_logit": torch.tensor([2.0, -2.0], requires_grad=True),
        "offset": torch.tensor([[1.0, 0.0, 0.0], [99.0, 99.0, 99.0]], requires_grad=True),
        "diameter_mm": torch.tensor([5.0, 99.0], requires_grad=True),
    }
    labels = torch.tensor([1.0, 0.0])
    offsets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sizes = torch.tensor([5.0, 0.0])
    losses = criterion(outputs, labels, offsets, sizes)
    assert torch.isfinite(losses["loss"])
    assert float(losses["offset_loss"]) < 1e-5
    assert float(losses["size_loss"]) < 1e-5
    print("[PASS] Offset/size losses ignore negative candidates")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 3 BASELINE MODEL UNIT TESTS")
    print("=" * 60)
    test_parameter_count()
    test_forward_backward_cpu()
    test_forward_backward_gpu()
    test_loss_positive_only_regression()
    print("=" * 60)
