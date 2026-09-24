"""
BEVFusionDetector: modelo principal de deteccao por fusao Camera-Radar em BEV.

Combina:
    1. Image Backbone (MobileNetV2) → features de imagem
    2. View Transformer (IPM) → projeta features para BEV
    3. Radar BEV Grid → features de radar
    4. Fusion Module → concatena e processa
    5. Detection Head → prediz mapa de segmentacao BEV

Fluxo:
    Image [B, 3, 384, 672]
        ↓ Backbone
    Features [B, 32, 24, 42]
        ↓ IPM Transformer
    Image BEV [B, 32, 128, 128]
        ↓
    ┌────────────────────────────────┐
    │ Fusion(Image BEV, Radar BEV)   │
    └────────────────────────────────┘
        ↓
    Fused [B, 64, 128, 128]
        ↓ Detection Head
    Seg Map [B, 1, 128, 128] (logits)
"""

import torch
import torch.nn as nn

from .backbones.image_backbone import ImageBackbone
from .modules.view_transformer import IPMTransformer
from .modules.fusion_module import FusionModule


class BEVFusionDetector(nn.Module):
    """
    Detector de obstaculos por fusao Camera-Radar em BEV.
    """

    def __init__(
        self,
        image_size: tuple = (384, 672),
        bev_size: tuple = (128, 128),
        bev_range: tuple = (-50.0, 50.0),
        backbone_out_channels: int = 32,
        radar_channels: int = 4,
        fusion_out_channels: int = 64,
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            image_size: (H, W) da imagem de entrada
            bev_size: (H, W) do grid BEV
            bev_range: (min_m, max_m) range em metros
            backbone_out_channels: Canais de saida do backbone
            radar_channels: Canais do grid radar
            fusion_out_channels: Canais apos fusao
            pretrained_backbone: Se True, usa pesos ImageNet
        """
        super().__init__()
        self.image_size = image_size
        self.bev_size = bev_size
        self.bev_range = bev_range

        # 1. Backbone de imagem
        self.backbone = ImageBackbone(
            out_channels=backbone_out_channels,
            pretrained=pretrained_backbone,
            freeze=False,
        )

        # 2. View Transformer (IPM)
        feat_h = image_size[0] // 16  # MobileNetV2 stride 16
        feat_w = image_size[1] // 16
        self.view_transformer = IPMTransformer(
            bev_size=bev_size,
            bev_range=bev_range,
            image_size=image_size,
            feature_size=(feat_h, feat_w),
        )

        # 3. Fusion Module
        self.fusion = FusionModule(
            img_channels=backbone_out_channels,
            radar_channels=radar_channels,
            out_channels=fusion_out_channels,
        )

        # 4. Detection Head (segmentacao binaria)
        self.detection_head = nn.Sequential(
            nn.Conv2d(fusion_out_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos do detection head."""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        image: torch.Tensor,
        radar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass completo.

        Args:
            image: Tensor[B, 3, H, W] — imagem RGB normalizada
            radar_bev: Tensor[B, 4, bev_h, bev_w] — grid radar BEV

        Returns:
            Tensor[B, 1, bev_h, bev_w] — logits de segmentacao BEV
        """
        # 1. Extrair features da imagem
        img_features = self.backbone(image)  # [B, 32, 24, 42]

        # 2. Transformar para BEV
        img_bev = self.view_transformer(img_features)  # [B, 32, 128, 128]

        # 3. Garantir mesmo tamanho entre img_bev e radar_bev
        if img_bev.shape[2:] != radar_bev.shape[2:]:
            radar_bev = nn.functional.interpolate(
                radar_bev,
                size=img_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # 4. Fusionar
        fused = self.fusion(img_bev, radar_bev)  # [B, 64, 128, 128]

        # 5. Detection head
        seg_logits = self.detection_head(fused)  # [B, 1, 128, 128]

        return seg_logits

    def predict(
        self,
        image: torch.Tensor,
        radar_bev: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Inferencia com threshold.

        Args:
            image: Tensor[B, 3, H, W]
            radar_bev: Tensor[B, 4, bev_h, bev_w]
            threshold: Limiar de confianca

        Returns:
            Tensor[B, 1, bev_h, bev_w] — mapa binario (0 ou 1)
        """
        logits = self.forward(image, radar_bev)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

    def count_parameters(self) -> dict:
        """Conta parametros por modulo."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        vt_params = sum(p.numel() for p in self.view_transformer.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.detection_head.parameters())
        total = backbone_params + vt_params + fusion_params + head_params

        return {
            "backbone": backbone_params,
            "view_transformer": vt_params,
            "fusion": fusion_params,
            "detection_head": head_params,
            "total": total,
        }


if __name__ == "__main__":
    # Teste completo
    model = BEVFusionDetector(
        image_size=(384, 672),
        bev_size=(128, 128),
        pretrained_backbone=False,
    )

    # Dados sinteticos
    image = torch.randn(2, 3, 384, 672)
    radar_bev = torch.randn(2, 4, 128, 128)

    # Forward
    seg_logits = model(image, radar_bev)
    print(f"Image:      {image.shape}")
    print(f"Radar BEV:  {radar_bev.shape}")
    print(f"Seg Logits: {seg_logits.shape}")

    # Predicao
    seg_pred = model.predict(image, radar_bev, threshold=0.5)
    print(f"Seg Pred:   {seg_pred.shape}")

    # Parametros
    params = model.count_parameters()
    print(f"\nParametros:")
    for k, v in params.items():
        print(f"  {k:20s}: {v/1e6:.2f}M")
