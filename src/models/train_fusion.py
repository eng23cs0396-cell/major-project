"""
Training Engine for Phase 5: Multimodal Dual-Branch Detector (ResNet3D + GATv2).

Fuses volumetric 3D image patches with arterial topology subgraphs using Cross-Attention,
supervised by Multi-Task Focal Loss with mixed precision (AMP) and Cosine Annealing LR.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from src.data.dataset import AneurysmPatchDataset
from src.data.multimodal_dataset import MultimodalAneurysmDataset, multimodal_collate_fn
from src.models.dual_branch_detector import DualBranchDetector
from src.models.losses import MultiTaskDetectionLoss
from src.models.trainer import (
    binary_auroc,
    classification_metrics,
    metrics_at_youden,
    set_seed,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _unpack_batch(batch: dict, device: torch.device):
    patches = batch["patch"].to(device, non_blocking=True)
    graphs = batch["graph"].to(device)
    labels = batch["label"].to(device, non_blocking=True)
    offsets = batch["offset"].to(device, non_blocking=True)
    sizes = batch["diameter_mm"].to(device, non_blocking=True)
    return patches, graphs, labels, offsets, sizes


def train_one_epoch(
    model: DualBranchDetector,
    loader: DataLoader,
    criterion: MultiTaskDetectionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "cls_loss": 0.0, "offset_loss": 0.0, "size_loss": 0.0}
    n_batches = 0

    for batch in loader:
        patches, graphs, labels, offsets, sizes = _unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(patches, graphs)
            losses = criterion(outputs, labels, offsets, sizes)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for key in running:
            running[key] += float(losses[key].detach().item())
        n_batches += 1

    denom = max(1, n_batches)
    return {k: v / denom for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: DualBranchDetector,
    loader: DataLoader,
    criterion: MultiTaskDetectionLoss,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    y_true: List[int] = []
    y_prob: List[float] = []
    size_err: List[float] = []
    n_batches = 0

    for batch in loader:
        patches, graphs, labels, offsets, sizes = _unpack_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(patches, graphs)
            losses = criterion(outputs, labels, offsets, sizes)

        running_loss += float(losses["loss"].detach().item())
        probs = outputs["aneurysm_prob"].detach().cpu().numpy().tolist()
        targets = labels.detach().cpu().numpy().tolist()
        y_prob.extend(probs)
        y_true.extend(targets)

        pos_mask = labels == 1
        if torch.any(pos_mask):
            pred_sz = outputs["diameter_mm"][pos_mask].detach().cpu().numpy()
            true_sz = sizes[pos_mask].detach().cpu().numpy()
            size_err.extend(np.abs(pred_sz - true_sz).tolist())
        n_batches += 1

    y_true_np = np.asarray(y_true, dtype=np.int32)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    metrics = metrics_at_youden(y_true_np, y_prob_np)
    metrics["val_loss"] = running_loss / max(1, n_batches)
    metrics["diameter_mae_mm"] = float(np.mean(size_err)) if size_err else float("nan")
    return metrics


def train_fusion(args: argparse.Namespace) -> Dict[str, float]:
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("random_seed", 42)))

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    use_amp = bool(cfg.get("training", {}).get("mixed_precision", True)) and (device.type == "cuda")

    # Datasets
    train_patch_ds = AneurysmPatchDataset(
        split_csv=args.train_split,
        eda_summary_csv=args.eda_csv,
        dataset_root=cfg["data"]["dataset_root"],
        neg_pos_ratio=args.neg_pos_ratio,
        jitter_range=int(cfg["data"].get("jitter_range", 2)),
        is_training=True,
        cache_patches=args.cache_patches,
    )
    val_patch_ds = AneurysmPatchDataset(
        split_csv=args.val_split,
        eda_summary_csv=args.eda_csv,
        dataset_root=cfg["data"]["dataset_root"],
        neg_pos_ratio=args.neg_pos_ratio,
        jitter_range=0,
        is_training=False,
        cache_patches=args.cache_patches,
    )

    if args.max_train_samples:
        train_patch_ds = Subset(train_patch_ds, list(range(min(args.max_train_samples, len(train_patch_ds)))))
    if args.max_val_samples:
        val_patch_ds = Subset(val_patch_ds, list(range(min(args.max_val_samples, len(val_patch_ds)))))

    # Wrap in MultimodalAneurysmDataset
    train_ds = MultimodalAneurysmDataset(train_patch_ds)
    val_ds = MultimodalAneurysmDataset(val_patch_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=0,  # Main process for Windows stability
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize Dual-Branch Detector
    model = DualBranchDetector(
        backbone=cfg.get("model", {}).get("cnn", {}).get("backbone", "resnet18_3d"),
        vis_dim=int(cfg.get("model", {}).get("cnn", {}).get("embedding_dim", 512)),
        topo_in_channels=int(cfg.get("model", {}).get("gat", {}).get("in_channels", 8)),
        topo_hidden_channels=int(cfg.get("model", {}).get("gat", {}).get("hidden_channels", 64)),
        topo_out_channels=int(cfg.get("model", {}).get("gat", {}).get("out_channels", 128)),
        topo_edge_dim=int(cfg.get("model", {}).get("gat", {}).get("edge_dim", 4)),
        topo_dim=int(cfg.get("model", {}).get("fusion", {}).get("fused_dim", 256)),
        fused_dim=int(cfg.get("model", {}).get("cnn", {}).get("embedding_dim", 512)),
        dropout=float(cfg.get("model", {}).get("fusion", {}).get("dropout", 0.2)),
        pretrained_baseline_path=args.pretrained_baseline,
    ).to(device)

    criterion = MultiTaskDetectionLoss(
        lambda_offset=args.lambda_offset,
        lambda_size=args.lambda_size,
        focal_alpha=args.focal_alpha,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history_path = os.path.join(args.checkpoint_dir, "fusion_train_history.json")
    best_auroc = -1.0
    best_metrics: Dict[str, float] = {}
    history: List[dict] = []
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        scheduler.last_epoch = int(ckpt["epoch"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            history = list(payload.get("history", []))
            best_metrics = dict(payload.get("best") or {})
            best_auroc = float(best_metrics.get("auroc", -1.0))
        print(f"Resumed from {args.resume} (starting at epoch {start_epoch})")

    patience = args.patience if args.patience is not None else int(cfg.get("training", {}).get("early_stopping_patience", 10))
    stale = 0

    print(
        f"[Phase 5 Fusion] Device: {device} | AMP: {use_amp} | "
        f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)} | "
        f"Epochs {start_epoch}-{args.epochs}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_stats = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": curr_lr,
            "train_loss": train_stats["loss"],
            "train_cls_loss": train_stats["cls_loss"],
            "train_offset_loss": train_stats["offset_loss"],
            "train_size_loss": train_stats["size_loss"],
            **val_stats,
        }
        history.append(epoch_record)

        auroc_val = val_stats.get("auroc", float("nan"))
        auroc_str = f"{auroc_val:.4f}" if np.isfinite(auroc_val) else "nan"
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['val_loss']:.4f} | "
            f"AUROC={auroc_str} | Sens@op={val_stats.get('sensitivity_op', 0.0):.3f} | "
            f"Spec@op={val_stats.get('specificity_op', 0.0):.3f} | diam_MAE={val_stats.get('diameter_mae_mm', 0.0):.3f} mm | "
            f"lr={curr_lr:.2e}"
        )

        # Save last checkpoint
        last_ckpt_path = os.path.join(args.checkpoint_dir, "last_fusion_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_stats,
                "config": cfg,
                "scaler_state_dict": scaler.state_dict(),
            },
            last_ckpt_path,
        )

        # Check for best checkpoint
        if np.isfinite(auroc_val) and auroc_val > best_auroc:
            best_auroc = auroc_val
            best_metrics = val_stats
            stale = 0
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_fusion_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_stats,
                    "config": cfg,
                    "scaler_state_dict": scaler.state_dict(),
                },
                best_ckpt_path,
            )
            print(f"  [SAVED] {best_ckpt_path} (AUROC: {best_auroc:.4f})")
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"Early stopping after {epoch} epochs (patience={patience}).")
                break

    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({"history": history, "best": best_metrics}, handle, indent=2)
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Train Multimodal Dual-Branch Detector")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--train-split", default="configs/train_split.csv")
    parser.add_argument("--val-split", default="configs/val_split.csv")
    parser.add_argument("--eda-csv", default="artifacts/eda/aneurysm_annotations_summary.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--pretrained-baseline", default="checkpoints/best_baseline_resnet3d.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-offset", type=float, default=0.1)
    parser.add_argument("--lambda-size", type=float, default=0.1)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--neg-pos-ratio", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-patches", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train_fusion(parse_args())
