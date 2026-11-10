"""
EarlyBEVFusionDetector: Modelo de detecção por Early Fusion Camera-Radar em BEV.

Diferença em relação ao Middle Fusion (BEVFusionDetector):
    - Middle Fusion:
        Câmera [B, 3, H, W] -> Backbone MobileNetV2 -> Features [B, 32, h, w] -> IPM -> Features BEV [B, 32, 128, 128]
        Radar [B, 10, N]    -> radar_to_bev_grid()                           -> Radar BEV [B, 4, 128, 128]
        Fusão: Concatena [32 + 4] canais em BEV -> Bloco Convolucional -> Detection Head

    - Early Fusion (este modelo):
        Câmera [B, 3, H, W] -> IPM direto na entrada -> Image BEV [B, 3, 128, 128]
        Radar [B, 10, N]    -> radar_to_bev_grid()  -> Radar BEV [B, 4, 128, 128]
        Fusão: Concatena canais brutos na entrada -> [B, 7, 128, 128] (3 RGB + 4 Radar)
        Backbone Unificado: MobileNetV2 adaptada para 7 canais processa a fusão desde a camada 0
        Decoder Multi-escala: Reconstrói o mapa de ocupação BEV [B, 1, 128, 128]
"""

import sys
import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.models.modules.view_transformer import IPMTransformer
else:
    from .modules.view_transformer import IPMTransformer


class EarlyBEVFusionBackbone(nn.Module):
    """
    Extrator de características multi-escala baseado no MobileNetV2 adaptado
    para entrada multimodal (7 canais: 3 RGB em BEV + 4 Radar em BEV).
    """

    def __init__(
        self,
        in_channels: int = 7,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        """
        Args:
            in_channels: Número de canais de entrada (3 RGB + 4 Radar = 7).
            pretrained: Se True, preserva os pesos pré-treinados do ImageNet
                        para os 3 canais de cor e inicializa os 4 canais de radar.
            freeze: Se True, congela os pesos do backbone para fine-tuning posterior.
        """
        super().__init__()
        self.in_channels = in_channels

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Adaptar a primeira camada convolucional para aceitar in_channels
        orig_conv = mobilenet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if pretrained:
                # Copiar os pesos do ImageNet para os primeiros 3 canais (RGB)
                new_conv.weight[:, :3, :, :] = orig_conv.weight
            else:
                nn.init.kaiming_normal_(new_conv.weight[:, :3, :, :], mode="fan_out", nonlinearity="relu")
            
            # Inicializar canais de radar (densidade, RCS, Vx, Vy) com Kaiming Normal
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")

        mobilenet.features[0][0] = new_conv

        # Estágios multi-escala da MobileNetV2 (features[0..13]):
        # Entrada: [B, 7, 128, 128]
        # Stage 1: stride 2  -> [B, 16, 64, 64]
        self.stage1 = nn.Sequential(*mobilenet.features[0:2])
        # Stage 2: stride 4  -> [B, 24, 32, 32]
        self.stage2 = nn.Sequential(*mobilenet.features[2:4])
        # Stage 3: stride 8  -> [B, 32, 16, 16]
        self.stage3 = nn.Sequential(*mobilenet.features[4:7])
        # Stage 4: stride 16 -> [B, 96, 8, 8]
        self.stage4 = nn.Sequential(*mobilenet.features[7:14])

        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna feature maps em 4 escalas diferentes para decodificação."""
        c1 = self.stage1(x)   # [B, 16, 64, 64]
        c2 = self.stage2(c1)  # [B, 24, 32, 32]
        c3 = self.stage3(c2)  # [B, 32, 16, 16]
        c4 = self.stage4(c3)  # [B, 96, 8, 8]
        return c1, c2, c3, c4

    def freeze(self):
        """Congela parâmetros do backbone."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Descongela parâmetros do backbone."""
        for param in self.parameters():
            param.requires_grad = True


class LightBEVDecoder(nn.Module):
    """
    Decodificador convolucional leve multi-escala com conexões laterais (estilo U-Net/FPN)
    para reconstruir a máscara BEV em resolução total 128x128.
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # Redução do canal mais profundo (C4: 96 ch -> 48 ch)
        self.lat4 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Fusão com C3 (48 + 32 = 80 ch -> 32 ch)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(48 + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Fusão com C2 (32 + 24 = 56 ch -> 32 ch)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(32 + 24, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Fusão com C1 (32 + 16 = 48 ch -> 32 ch)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Cabeça de predição final (upsample 64x64 -> 128x128)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        p4 = self.lat4(c4)  # [B, 48, 8, 8]
        p4_up = F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)  # [B, 48, 16, 16]

        p3 = self.fuse3(torch.cat([p4_up, c3], dim=1))  # [B, 32, 16, 16]
        p3_up = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)  # [B, 32, 32, 32]

        p2 = self.fuse2(torch.cat([p3_up, c2], dim=1))  # [B, 32, 32, 32]
        p2_up = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)  # [B, 32, 64, 64]

        p1 = self.fuse1(torch.cat([p2_up, c1], dim=1))  # [B, 32, 64, 64]
        p1_up = F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False)  # [B, 32, 128, 128]

        logits = self.head(p1_up)  # [B, 1, 128, 128]
        return logits


class EarlyBEVFusionDetector(nn.Module):
    """
    Detector de obstáculos multimodal Camera-Radar baseado em Early Fusion no espaço BEV.

    Compatibilidade:
        Possui a mesma interface de chamada de BEVFusionDetector:
            logits = model(image, radar_bev)
        Não exige alteração no pipeline de dados ou reprocessamento no Google Colab.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (384, 672),
        bev_size: Tuple[int, int] = (128, 128),
        bev_range: Tuple[float, float] = (-50.0, 50.0),
        radar_channels: int = 4,
        out_channels: int = 1,
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            image_size: (H, W) da imagem frontal da câmera.
            bev_size: (H, W) da grade métrica BEV.
            bev_range: (min_m, max_m) alcance em metros nos eixos X e Y.
            radar_channels: número de canais do grid de radar (padrão: 4).
            out_channels: número de classes/canais de saída (1 para segmentação binária).
            pretrained_backbone: se True, utiliza pesos pré-treinados do ImageNet.
        """
        super().__init__()
        self.image_size = image_size
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.radar_channels = radar_channels

        # 1. View Transformer (IPM direto na imagem RGB frontal)
        # Transforma [B, 3, 384, 672] diretamente em [B, 3, 128, 128]
        self.image_view_transformer = IPMTransformer(
            bev_size=bev_size,
            bev_range=bev_range,
            image_size=image_size,
            feature_size=image_size,
        )

        # 2. Backbone Unificado (MobileNetV2 com 3 RGB + 4 Radar = 7 canais)
        in_channels = 3 + radar_channels
        self.backbone = EarlyBEVFusionBackbone(
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            freeze=False,
        )

        # 3. Decodificador Multi-escala
        self.decoder = LightBEVDecoder(out_channels=out_channels)

    def forward(
        self,
        image: torch.Tensor,
        radar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass de Early Fusion.

        Args:
            image: Tensor[B, 3, 384, 672] — imagem frontal normalizada.
            radar_bev: Tensor[B, 4, 128, 128] — grid BEV de radar (densidade, RCS, Vx, Vy).

        Returns:
            Tensor[B, 1, 128, 128] — logits de segmentação BEV.
        """
        # 1. Projeção direta da imagem para o espaço BEV (Early IPM)
        img_bev = self.image_view_transformer(image)  # [B, 3, 128, 128]

        # 2. Garantir compatibilidade espacial
        if img_bev.shape[2:] != radar_bev.shape[2:]:
            radar_bev = F.interpolate(
                radar_bev,
                size=img_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # 3. Early Fusion: concatenação dos dados pré-processados na entrada
        fused_input = torch.cat([img_bev, radar_bev], dim=1)  # [B, 7, 128, 128]

        # 4. Extração conjunta de características via MobileNetV2
        c1, c2, c3, c4 = self.backbone(fused_input)

        # 5. Decodificação em alta resolução BEV
        seg_logits = self.decoder(c1, c2, c3, c4)  # [B, 1, 128, 128]

        return seg_logits

    def predict(
        self,
        image: torch.Tensor,
        radar_bev: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predição binarizada para inferência."""
        logits = self.forward(image, radar_bev)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    def count_parameters(self) -> Dict[str, int]:
        """Conta parâmetros por componente."""
        vt_params = sum(p.numel() for p in self.image_view_transformer.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = vt_params + backbone_params + decoder_params

        return {
            "view_transformer": vt_params,
            "backbone": backbone_params,
            "decoder": decoder_params,
            "total": total,
        }


if __name__ == "__main__":
    model = EarlyBEVFusionDetector(pretrained_backbone=False)
    img = torch.randn(2, 3, 384, 672)
    rad = torch.randn(2, 4, 128, 128)
    out = model(img, rad)
    print("EarlyBEVFusionDetector output:", out.shape)
    pred = model.predict(img, rad)
    print("EarlyBEVFusionDetector pred:", pred.shape)
    params = model.count_parameters()
    for k, v in params.items():
        print(f"  {k:20s}: {v/1e6:.2f}M")
