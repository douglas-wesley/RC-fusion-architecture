"""
Evaluate: metricas de avaliacao para segmentacao BEV.

Metricas implementadas:
    - IoU (Intersection over Union) por classe
    - mIoU (mean IoU)
    - Precision, Recall, F1
    - BEV Accuracy

Uso:
    python -m src.engine.evaluate --config config/default.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
from typing import Dict, List

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset.nuscenes_dataset import NuScenesFusionDataset, collate_fn
from src.dataset.radar_transforms import radar_to_bev_grid
from src.models.detector import BEVFusionDetector
from src.engine.train import load_config, build_model


class BEVMetrics:
    """Acumula metricas de segmentacao BEV."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reseta contadores."""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives
        self.total_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Atualiza metricas com um batch."""
        pred_bin = (pred > self.threshold).float()
        target_bin = (target > self.threshold).float()

        self.tp += ((pred_bin == 1) & (target_bin == 1)).sum().item()
        self.fp += ((pred_bin == 1) & (target_bin == 0)).sum().item()
        self.fn += ((pred_bin == 0) & (target_bin == 1)).sum().item()
        self.tn += ((pred_bin == 0) & (target_bin == 0)).sum().item()
        self.total_pixels += target.numel()

    def compute(self) -> Dict[str, float]:
        """Computa metricas finais."""
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (self.tp + self.tn) / (self.total_pixels + 1e-6)

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


@torch.no_grad()
def evaluate(
    model: BEVFusionDetector,
    dataloader: DataLoader,
    device: torch.device,
    cfg: dict,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Avalia o modelo no dataset de validacao."""
    model.eval()
    metrics = BEVMetrics(threshold=threshold)

    for batch in dataloader:
        image = batch["image"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_mask = batch["radar_mask"].to(device)
        seg_gt = batch["bev_segmentation"].to(device)

        # Radar points → BEV grid
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

        # Forward
        seg_logits = model(image, radar_bev)
        seg_prob = torch.sigmoid(seg_logits)

        # Atualizar metricas
        metrics.update(seg_prob, seg_gt)

    return metrics.compute()


def print_metrics(metrics: Dict[str, float]):
    """Imprime metricas formatadas."""
    print("\n" + "=" * 50)
    print("  METRICAS DE AVALIACAO — BEV Segmentation")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f} ({value*100:.1f}%)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Avaliar BEVFusionDetector")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo
    model = build_model(cfg, device)
    epoch, _ = load_checkpoint(model, None, args.checkpoint, device)
    print(f"Checkpoint carregado: epoch {epoch}")

    # Dataloader
    val_dataset = NuScenesFusionDataset(
        dataroot=cfg["data"]["dataroot"],
        version=cfg["data"]["version"],
        split="val",
        train_ratio=cfg["data"]["train_ratio"],
        image_size=tuple(cfg["data"]["image_size"]),
        bev_size=tuple(cfg["data"]["bev_size"]),
        bev_range=tuple(cfg["data"]["bev_range"]),
        radar_max_points=cfg["data"]["radar_max_points"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Avaliar
    metrics = evaluate(model, val_loader, device, cfg, args.threshold)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
