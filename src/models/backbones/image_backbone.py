"""
Image Backbone para extracao de features de imagem.

Utiliza MobileNetV2 pre-treinado no ImageNet como extrator de features leve.
A saida e um feature map 2D em alta resolucao, pronto para ser transformado
em BEV pelo View Transformer.

Arquitetura:
    Input [B, 3, 384, 672]
        ↓
    MobileNetV2 features[0..13] (stride 16)
        ↓
    Feature Map [B, 64, 24, 42]
        ↓
    Conv 1x1 (reduz canais)
        ↓
    Output [B, out_channels, 24, 42]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageBackbone(nn.Module):
    """
    Extrator de features de imagem baseado no MobileNetV2.

    Remove as ultimas camadas (features[14..18] + classifier) e mantem
    apenas as features de stride 16 (resolucao 24x42 para input 384x672).
    """

    def __init__(
        self,
        out_channels: int = 32,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        """
        Args:
            out_channels: Numero de canais de saida (default 32)
            pretrained: Se True, carrega pesos do ImageNet
            freeze: Se True, congela todos os parametros (backbone treinavel depois)
        """
        super().__init__()
        self.out_channels = out_channels

        # Carregar MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        else:
            weights = None

        mobilenet = models.mobilenet_v2(weights=weights)

        # Extrair features[0..13] → stride 16, 96 canais
        self.features = nn.Sequential(*list(mobilenet.features.children())[:14])

        # Reduzir canais de 96 → out_channels com conv 1x1
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Congelar se solicitado
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Inicializacao das pesos da conv 1x1
        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos da camada de reducao de canais."""
        for m in self.channel_reduce.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, 3, H, W] — imagem normalizada

        Returns:
            Tensor[B, out_channels, H/16, W/16] — feature map reduzido
        """
        x = self.features(x)       # [B, 64, 24, 42]
        x = self.channel_reduce(x)  # [B, out_channels, 24, 42]
        return x

    def get_output_shape(self, input_size: tuple = (384, 672)) -> tuple:
        """
        Retorna a forma do output sem precisar rodar um forward pass.
        Util para configurar o View Transformer.
        """
        h, w = input_size
        # MobileNetV2 stride 16
        out_h = h // 16
        out_w = w // 16
        return (self.out_channels, out_h, out_w)

    def unfreeze(self):
        """Descongela o backbone para fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze(self):
        """Congela o backbone."""
        for param in self.features.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # Teste rapido
    model = ImageBackbone(out_channels=32, pretrained=False)
    x = torch.randn(2, 3, 384, 672)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Output shape (method): {model.get_output_shape()}")
