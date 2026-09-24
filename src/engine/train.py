"""
Training loop para BEVFusionDetector.

Suporta:
    - Mixed precision (AMP) para economia de memoria
    - Learning rate scheduling (cosine ou step)
    - Checkpointing periodico
    - Early stopping
    - TensorBoard logging
    - Freeze/unfreeze do backbone

Uso:
    python -m src.engine.train --config config/default.yaml
    python -m src.engine.train --config config/default.yaml --dataroot /content/data/sets/nuscenes
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset.nuscenes_dataset import NuScenesFusionDataset, collate_fn
from src.dataset.radar_transforms import radar_to_bev_grid
from src.models.detector import BEVFusionDetector


def load_config(config_path: str) -> dict:
    """Carrega configuracao YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Define seed para reprodutibilidade."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> BEVFusionDetector:
    """Constroi o modelo a partir do config."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = BEVFusionDetector(
        image_size=tuple(data_cfg["image_size"]),
        bev_size=tuple(data_cfg["bev_size"]),
        bev_range=tuple(data_cfg["bev_range"]),
        backbone_out_channels=model_cfg["fusion"]["img_channels"],
        radar_channels=data_cfg["radar_input_channels"],
        fusion_out_channels=model_cfg["fusion"]["out_channels"],
        pretrained_backbone=model_cfg["backbone"]["pretrained"],
    )

    return model.to(device)


def build_dataloaders(cfg: dict, dataroot: str):
    """Constroi dataloaders de treino e validacao."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    train_dataset = NuScenesFusionDataset(
        dataroot=dataroot,
        version=data_cfg["version"],
        split="train",
        train_ratio=data_cfg["train_ratio"],
        image_size=tuple(data_cfg["image_size"]),
        bev_size=tuple(data_cfg["bev_size"]),
        bev_range=tuple(data_cfg["bev_range"]),
        radar_max_points=data_cfg["radar_max_points"],
    )

    val_dataset = NuScenesFusionDataset(
        dataroot=dataroot,
        version=data_cfg["version"],
        split="val",
        train_ratio=data_cfg["train_ratio"],
        image_size=tuple(data_cfg["image_size"]),
        bev_size=tuple(data_cfg["bev_size"]),
        bev_range=tuple(data_cfg["bev_range"]),
        radar_max_points=data_cfg["radar_max_points"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: dict):
    """Constroi optimizer e scheduler."""
    train_cfg = cfg["training"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["step_size"],
            gamma=train_cfg["step_gamma"],
        )

    return optimizer, scheduler


def compute_loss(
    seg_logits: torch.Tensor,
    seg_gt: torch.Tensor,
    radar_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calcula BCE loss para segmentacao binaria BEV.
    Aplica ponderacao para lidar com desbalanceamento (poucos pixels positivos).
    """
    # BCE com logits
    bce = nn.functional.binary_cross_entropy_with_logits(
        seg_logits, seg_gt, reduction="none"
    )

    # Ponderacao: pixels positivos recebem peso maior
    pos_weight = torch.tensor([5.0], device=seg_logits.device)
    pos_mask = seg_gt > 0.5
    neg_mask = ~pos_mask

    loss = (bce * pos_mask.float()).sum() * pos_weight + (bce * neg_mask.float()).sum()
    loss = loss / seg_gt.numel()

    return loss


def train_one_epoch(
    model: BEVFusionDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Treina por uma epoch. Retorna metricas."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_mask = batch["radar_mask"].to(device)
        seg_gt = batch["bev_segmentation"].to(device)

        # Converter radar points para BEV grid
        B = image.shape[0]
        radar_bev_list = []
        for i in range(B):
            bev = radar_to_bev_grid(
                radar_points[i],
                radar_mask[i],
                bev_size=tuple(cfg["data"]["bev_size"]),
                bev_range=tuple(cfg["data"]["bev_range"]),
            )
            radar_bev_list.append(bev)
        radar_bev = torch.stack(radar_bev_list, dim=0).to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with autocast():
                seg_logits = model(image, radar_bev)
                loss = compute_loss(seg_logits, seg_gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits = model(image, radar_bev)
            loss = compute_loss(seg_logits, seg_gt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: BEVFusionDetector,
    dataloader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    """Valida o modelo. Retorna metricas."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in dataloader:
        image = batch["image"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_mask = batch["radar_mask"].to(device)
        seg_gt = batch["bev_segmentation"].to(device)

        # Converter radar points para BEV grid
        B = image.shape[0]
        radar_bev_list = []
        for i in range(B):
            bev = radar_to_bev_grid(
                radar_points[i],
                radar_mask[i],
                bev_size=tuple(cfg["data"]["bev_size"]),
                bev_range=tuple(cfg["data"]["bev_range"]),
            )
            radar_bev_list.append(bev)
        radar_bev = torch.stack(radar_bev_list, dim=0).to(device)

        seg_logits = model(image, radar_bev)
        loss = compute_loss(seg_logits, seg_gt)

        # Calcular IoU
        seg_pred = (torch.sigmoid(seg_logits) > 0.5).float()
        intersection = (seg_pred * seg_gt).sum()
        union = seg_pred.sum() + seg_gt.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()

        total_loss += loss.item()
        total_iou += iou
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_iou": total_iou / max(n_batches, 1),
    }


def save_checkpoint(
    model: BEVFusionDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Salva checkpoint do treinamento."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: BEVFusionDetector,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device,
):
    """Carrega checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def train(config_path: str, dataroot: Optional[str] = None):
    """Funcao principal de treinamento."""
    global cfg

    cfg = load_config(config_path)

    # Override dataroot se fornecido
    if dataroot:
        cfg["data"]["dataroot"] = dataroot

    # Seed
    set_seed(cfg["training"]["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Modelo
    model = build_model(cfg, device)
    params = model.count_parameters()
    print(f"Parametros totais: {params['total']/1e6:.2f}M")

    # Dataloaders
    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["dataroot"])
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # Optimizer e scheduler
    optimizer, scheduler = build_optimizer(model, cfg)

    # AMP scaler
    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # TensorBoard (opcional, com fallback seguro caso o ambiente tenha conflito TensorFlow/JAX)
    writer = None
    if cfg["training"]["use_tensorboard"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = cfg["training"]["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard ativado em: {log_dir}")
        except Exception as e:
            print(f"AVISO: TensorBoard desativado devido a conflito de ambiente: {e}")
            writer = None

    # Checkpoint dir
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    # ── Loop de treinamento ────────────────────────────────────────────
    n_epochs = cfg["training"]["epochs"]
    print(f"\nIniciando treinamento por {n_epochs} epochs...")

    for epoch in range(n_epochs):
        start_time = time.time()

        # Freeze backbone nas primeiras N epochs
        if epoch < cfg["model"]["backbone"]["freeze_epochs"]:
            model.backbone.freeze()
        else:
            model.backbone.unfreeze()

        # Treinar
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp
        )

        # Validar
        val_metrics = validate(model, val_loader, device, cfg)

        # Scheduler
        scheduler.step()

        # Tempo
        elapsed = time.time() - start_time

        # Log
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{n_epochs}] "
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"Val Loss: {val_metrics['val_loss']:.4f} "
            f"IoU: {val_metrics['val_iou']:.4f} "
            f"LR: {lr:.2e} "
            f"Time: {elapsed:.1f}s"
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar("Metrics/IoU", val_metrics["val_iou"], epoch)
            writer.add_scalar("LR", lr, epoch)

        # Checkpoint periodico
        if (epoch + 1) % cfg["training"]["save_every_n_epochs"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)

        # Melhor modelo
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            print(f"  → Melhor modelo salvo ({best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping na epoch {epoch+1} (paciência: {patience})")
                break

    # Salvar ultimo modelo
    last_path = os.path.join(ckpt_dir, "last_model.pth")
    save_checkpoint(model, optimizer, epoch, val_metrics, last_path)

    if writer is not None:
        writer.close()

    print("\nTreinamento concluído!")
    print(f"Melhor val loss: {best_val_loss:.4f}")
    print(f"Checkpoints salvos em: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar BEVFusionDetector")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Caminho do arquivo de configuracao",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
        help="Override do caminho do dataset (opcional)",
    )
    args = parser.parse_args()

    train(args.config, args.dataroot)
