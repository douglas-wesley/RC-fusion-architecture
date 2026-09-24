"""
Fusion Module: combina features de imagem BEV e radar BEV.

Realiza a fusao multimodal (Middle Fusion) concatenando os features
de imagem transformados para BEV com o grid de radar BEV,
seguido de camadas convolucionais para integrar as informacoes.

Arquitetura:
    Image BEV [B, img_ch, H, W] ─┐
                                  ├→ Concat → [B, img_ch+radar_ch, H, W]
    Radar BEV [B, radar_ch, H, W]─┘
                                       ↓
                                  Conv 3x3 → BN → ReLU
                                       ↓
                                  Conv 3x3 → BN → ReLU
                                       ↓
                                  Output [B, out_ch, H, W]
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Modulo de fusao para features BEV de imagem e radar.
    """

    def __init__(
        self,
        img_channels: int = 32,
        radar_channels: int = 4,
        out_channels: int = 64,
    ):
        """
        Args:
            img_channels: Canais do feature map de imagem em BEV
            radar_channels: Canais do grid radar BEV
            out_channels: Canais de saida apos fusao
        """
        super().__init__()
        in_channels = img_channels + radar_channels

        self.fusion_layers = nn.Sequential(
            # Primeira camada de fusao
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Segunda camada de refinamento
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Inicializacao He para convolucoes."""
        for m in self.fusion_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        img_bev: torch.Tensor,
        radar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatena e processa features de imagem e radar em BEV.

        Args:
            img_bev:  Tensor[B, img_channels, H, W] — features de imagem em BEV
            radar_bev: Tensor[B, radar_channels, H, W] — grid radar em BEV

        Returns:
            Tensor[B, out_channels, H, W] — features fusionadas
        """
        # Concatenar ao longo do eixo de canais
        fused = torch.cat([img_bev, radar_bev], dim=1)  # [B, img+radar, H, W]

        # Processar com convolucoes
        out = self.fusion_layers(fused)

        return out


if __name__ == "__main__":
    # Teste rapido
    model = FusionModule(img_channels=32, radar_channels=4, out_channels=64)
    img = torch.randn(2, 32, 128, 128)
    radar = torch.randn(2, 4, 128, 128)
    out = model(img, radar)
    print(f"Image BEV:  {img.shape}")
    print(f"Radar BEV:  {radar.shape}")
    print(f"Output:     {out.shape}")
    print(f"Params:     {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
