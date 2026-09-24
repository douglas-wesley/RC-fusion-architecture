"""
View Transformer: converte features de Visao Frontal para BEV.

Utiliza Inverse Perspective Mapping (IPM) para projetar o feature map
da imagem (perspectiva frontal) para o plano BEV (vista de cima).

Metodo:
    1. Gerar grid de homografia baseado nas matrizes de calibração
    2. Usar grid_sample do PyTorch para warpear os features
    3. Retornar feature map em coordenadas BEV

Referencia:
    Fast and furious: Real time end-to-end 3d detection, tracking and
    motion forecasting with a single convolutional network. IEEE IV 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IPMTransformer(nn.Module):
    """
    Transforma features de perspectiva frontal para BEV usando IPM.

    Assume que o chao e plano (Z=0) e utiliza a homografia inversa
    para mapear pixels da imagem de volta para coordenadas do mundo.
    """

    def __init__(
        self,
        bev_size: tuple = (128, 128),
        bev_range: tuple = (-50.0, 50.0),
        image_size: tuple = (384, 672),
        feature_size: tuple = (24, 42),
    ):
        """
        Args:
            bev_size: (H, W) do grid BEV de saida
            bev_range: (min_m, max_m) range em metros
            image_size: (H, W) da imagem original
            feature_size: (H, W) do feature map do backbone
        """
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.bev_min, self.bev_max = bev_range
        self.image_h, self.image_w = image_size
        self.feat_h, self.feat_w = feature_size

        # Pre-computar grid de homografia (constante, nao treinavel)
        grid = self._compute_ipm_grid()
        self.register_buffer("grid", grid)

    def _compute_ipm_grid(self) -> torch.Tensor:
        """
        Computa o grid de homografia para IPM.

        Mapeia cada pixel do BEV grid para o pixel correspondente
        no feature map da imagem, assumindo chao plano.

        Returns:
            Tensor[1, bev_h, bev_w, 2] — grid normalizado [-1, 1]
        """
        # Coordenadas do BEV grid em metros
        bev_min, bev_max = self.bev_min, self.bev_max
        x_coords = torch.linspace(bev_min, bev_max, self.bev_w)
        y_coords = torch.linspace(bev_max, bev_min, self.bev_h)  # invertido para Y apontar para frente

        # Grid 3D no plano do chao (Z=0)
        # BEV: x = lateral, y = frente (nuScenes convention)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing="xy")
        zz = torch.zeros_like(xx)
        bev_points = torch.stack([xx, yy, zz], dim=-1)  # [bev_h, bev_w, 3]

        # Parametros da câmera (estimados para CAM_FRONT do nuScenes)
        # Estes valores serao sobrescritos pelo calibration do dataset
        # Valores padrao approximados para CAM_FRONT
        focal_length_x = self.image_w * 0.8  # approx
        focal_length_y = self.image_h * 0.8
        cx = self.image_w / 2.0
        cy = self.image_h / 2.0

        # Altura da camera acima do chao (~1.5m para CAM_FRONT nuScenes)
        cam_height = 1.5
        # Angulo de inclinacao da camera (pitch) ~ graus para baixo
        pitch = -0.1  # radianos (levemente apontando para baixo)

        # Matriz de rotacao da camera (apenas pitch)
        cos_pitch = torch.cos(torch.tensor(pitch))
        sin_pitch = torch.sin(torch.tensor(pitch))
        R = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, sin_pitch],
            [0.0, -sin_pitch, cos_pitch],
        ])

        # Projetar BEV points para a imagem
        # Transformar para coordenadas da camera
        # camera_coords = R @ (bev_point - camera_position)
        cam_pos = torch.tensor([0.0, 0.0, cam_height])
        bev_homo = torch.cat([
            bev_points.reshape(-1, 3),
            torch.ones(self.bev_h * self.bev_w, 1)
        ], dim=1)  # [bev_h*bev_w, 3] → [bev_h*bev_w, 4]

        # Transformacao basica: projecao perspectiva simplificada
        # Para IPM, usamos a relacao: u = fx * (x/z) + cx, v = fy * (y/z) + cy
        # Com chao plano: z = cam_height, y = dist_frente, x = lateral

        # Coordenadas normalizadas no BEV
        bev_x = bev_points[:, :, 0].reshape(-1)  # lateral
        bev_y = bev_points[:, :, 1].reshape(-1)  # frente

        # Projetar para imagem (simplificado para chao plano)
        # u = fx * (x / y) + cx  (x = lateral, y = distancia)
        # v = fy * (cam_height / y) + cy  (altura projetada)

        # Evitar divisao por zero
        bev_y_safe = torch.clamp(bev_y, min=0.1)

        u = focal_length_x * (bev_x / bev_y_safe) + cx
        v = focal_length_y * (cam_height / bev_y_safe) + cy

        # Normalizar para [-1, 1] (grid_sample format)
        u_norm = 2.0 * (u / self.image_w) - 1.0
        v_norm = 2.0 * (v / self.image_h) - 1.0

        grid = torch.stack([u_norm, v_norm], dim=-1)  # [bev_h*bev_w, 2]
        grid = grid.reshape(1, self.bev_h, self.bev_w, 2)

        return grid

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transforma features de perspectiva para BEV.

        Args:
            features: Tensor[B, C, feat_h, feat_w] — features do backbone

        Returns:
            Tensor[B, C, bev_h, bev_w] — features em BEV
        """
        B, C, feat_h, feat_w = features.shape

        # Expandir grid para o batch
        grid = self.grid.expand(B, -1, -1, -1)

        # Grid sample: warpear features conforme homografia
        bev_features = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return bev_features

    def update_calibration(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        cam_height: float = 1.5,
    ):
        """
        Atualiza a homografia com calibragem real da camera.
        Deve ser chamado antes do forward se as calibragens estiverem disponiveis.

        Args:
            intrinsic: Matriz intrinseca 3x3 da camera
            extrinsic: Matriz extrinseca 4x4 (camera→world)
            cam_height: Altura da camera em metros
        """
        # Recalcula o grid com calibragem real
        # Implementacao futura: usar intrinsic e extrinsic reais
        pass


class MLPTransformer(nn.Module):
    """
    Transformador baseado em MLP (alternativa ao IPM).

    Aprende a transformacao perspective→BEV de forma aprendivel.
    Mais flexivel que IPM, mas precisa de mais dados para treinar.
    """

    def __init__(
        self,
        in_channels: int = 32,
        bev_size: tuple = (128, 128),
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_size

        self.transform = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(in_channels * 24 * 42, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, in_channels * bev_size[0] * bev_size[1]),
            nn.ReLU(inplace=True),
        )
        self.out_channels = in_channels
        self.bev_size = bev_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        x = self.transform(features)
        x = x.view(B, self.out_channels, self.bev_h, self.bev_w)
        return x


if __name__ == "__main__":
    # Teste rapido
    model = IPMTransformer(
        bev_size=(128, 128),
        bev_range=(-50.0, 50.0),
        image_size=(384, 672),
        feature_size=(24, 42),
    )
    x = torch.randn(2, 32, 24, 42)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Grid shape: {model.grid.shape}")
