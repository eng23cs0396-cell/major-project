"""
Mixed-precision trainer for the Phase 3 3D ResNet baseline.

Usage:
  .venv\\Scripts\\python -m src.models.trainer
  .venv\\Scripts\\python -m src.models.trainer --epochs 1 --max-train-samples 8 --max-val-samples 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset import AneurysmPatchDataset
from src.models.losses import MultiTaskDetectionLoss
from src.models.resnet3d import build_baseline_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        "auroc": binary_auroc(y_true, y_prob),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_subset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _unpack_batch(batch: dict, device: torch.device):
    patches = batch["patch"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    offsets = batch["offset_voxel"].to(device, non_blocking=True)
    sizes = batch["size_mm"].to(device, non_blocking=True)
    return patches, labels, offsets, sizes


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "cls_loss": 0.0, "offset_loss": 0.0, "size_loss": 0.0}
    n_batches = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="Train", leave=False):
        patches, labels, offsets, sizes = _unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(patches)
            losses = criterion(outputs, labels, offsets, sizes)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        for key in running:
            running[key] += float(losses[key if key != "loss" else "loss"].detach().cpu())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    y_true: List[float] = []
    y_prob: List[float] = []
    size_err: List[float] = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="Val", leave=False):
        patches, labels, offsets, sizes = _unpack_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(patches)
            losses = criterion(outputs, labels, offsets, sizes)
        running_loss += float(losses["loss"].detach().cpu())
        n_batches += 1
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(outputs["aneurysm_prob"].float().detach().cpu().numpy().tolist())
        pos = labels > 0
        if pos.any():
            err = torch.abs(outputs["diameter_mm"][pos] - sizes[pos])
            size_err.extend(err.float().detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int32)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    metrics = classification_metrics(y_true_np, y_prob_np)
    metrics["val_loss"] = running_loss / max(1, n_batches)
    metrics["diameter_mae_mm"] = float(np.mean(size_err)) if size_err else float("nan")
    return metrics


def run_training(args: argparse.Namespace) -> Dict[str, float]:
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("random_seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    use_amp = bool(cfg.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"

    train_ds = AneurysmPatchDataset(
        split_csv=args.train_split,
        eda_summary_csv=args.eda_csv,
        dataset_root=args.dataset_root,
        is_training=True,
        neg_pos_ratio=args.neg_pos_ratio,
        cache_patches=args.cache_patches,
    )
    val_ds = AneurysmPatchDataset(
        split_csv=args.val_split,
        eda_summary_csv=args.eda_csv,
        dataset_root=args.dataset_root,
        is_training=False,
        neg_pos_ratio=args.neg_pos_ratio,
        cache_patches=args.cache_patches,
    )
    train_ds = maybe_subset(train_ds, args.max_train_samples)
    val_ds = maybe_subset(val_ds, args.max_val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_cfg = cfg.get("model", {}).get("cnn", {})
    model = build_baseline_model(
        backbone=model_cfg.get("backbone", "resnet18_3d"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    criterion = MultiTaskDetectionLoss(lambda_offset=args.lambda_offset, lambda_size=args.lambda_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auroc = -1.0
    best_metrics: Dict[str, float] = {}
    history = []
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 10))
    stale = 0

    print(
        f"Device: {device} | AMP: {use_amp} | Cache: {args.cache_patches} | "
        f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_stats = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "lr": curr_lr,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **val_stats,
        }
        history.append(row)
        auroc = val_stats.get("auroc", float("nan"))
        auroc_str = f"{auroc:.4f}" if auroc == auroc else "n/a"
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['val_loss']:.4f} | "
            f"AUROC={auroc_str} | Sens={val_stats['sensitivity']:.3f} | "
            f"Spec={val_stats['specificity']:.3f} | diam_MAE={val_stats['diameter_mae_mm']:.3f} mm | "
            f"lr={curr_lr:.2e}"
        )

        finite_auroc = auroc == auroc
        if not best_metrics:
            improved = True
        elif finite_auroc and auroc > best_auroc:
            improved = True
        elif not finite_auroc and val_stats["val_loss"] < best_metrics.get("val_loss", float("inf")):
            improved = True
        else:
            improved = False

        # Always save last checkpoint
        last_ckpt_path = os.path.join(args.checkpoint_dir, "last_baseline_resnet3d.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_stats,
                "config": cfg,
            },
            last_ckpt_path,
        )

        if improved:
            best_auroc = auroc if finite_auroc else best_auroc
            best_metrics = val_stats
            stale = 0
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_baseline_resnet3d.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_stats,
                    "config": cfg,
                },
                best_ckpt_path,
            )
            print(f"  saved {best_ckpt_path}")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs (patience={patience}).")
                break

    history_path = os.path.join(args.checkpoint_dir, "baseline_train_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({"history": history, "best": best_metrics}, handle, indent=2)
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 3 3D ResNet baseline")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--dataset-root", default="dataset(topAneu)")
    parser.add_argument("--train-split", default="configs/train_split.csv")
    parser.add_argument("--val-split", default="configs/val_split.csv")
    parser.add_argument("--eda-csv", default="artifacts/eda/aneurysm_annotations_summary.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--neg-pos-ratio", type=float, default=1.0)
    parser.add_argument("--lambda-offset", type=float, default=1.0)
    parser.add_argument("--lambda-size", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-patches", action="store_true", default=True, help="Cache patches in RAM")
    parser.add_argument("--no-cache-patches", action="store_false", dest="cache_patches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    if args.epochs is None:
        args.epochs = int(train_cfg.get("epochs", 50))
    if args.batch_size is None:
        args.batch_size = int(train_cfg.get("batch_size", 4))
    if args.lr is None:
        args.lr = float(train_cfg.get("learning_rate", 1e-4))
    if args.num_workers is None:
        args.num_workers = int(train_cfg.get("num_workers", 0))

    # On Windows, num_workers=0 avoids process spawn overhead when in-memory cache is used
    if os.name == "nt" and args.cache_patches and args.num_workers > 0:
        args.num_workers = 0

    return args


if __name__ == "__main__":
    run_training(parse_args())
