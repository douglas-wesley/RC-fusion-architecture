"""
Transformacoes de radar para grid BEV (Bird's-Eye View).

Converte nuvens de pontos radar esparsas em grids 2D densos
multiplos canais, otimizados para CNNs.

Formato do grid de saida:
    Canal 0: Densidade (count de pontos por celula)
    Canal 1: RCS medio (tamanho/refletividade do alvo)
    Canal 2: Velocidade X media (frente/tras)
    Canal 3: Velocidade Y media (esquerda/direita)
"""

import numpy as np
import torch


def radar_to_bev_grid(
    radar_points: torch.Tensor,
    radar_mask: torch.Tensor,
    bev_size: tuple = (128, 128),
    bev_range: tuple = (-50.0, 50.0),
    n_channels: int = 4,
) -> torch.Tensor:
    """
    Converte pontos radar (padded) em um grid BEV denso multi-canal.

    Args:
        radar_points: Tensor[10, max_pts] — x, y, z, rcs, vx, vy, vx_comp, vy_comp, dist, azimuth
        radar_mask:   Tensor[max_pts]     — 1=pt real, 0=padding
        bev_size:     (H, W) do grid de saida
        bev_range:    (min_m, max_m) range em metros
        n_channels:   Numero de canais de saida (4 por padrao)

    Returns:
        Tensor[n_channels, bev_h, bev_w] — grid BEV denso
    """
    bev_h, bev_w = bev_size
    bev_min, bev_max = bev_range
    device = radar_points.device

    grid = torch.zeros((n_channels, bev_h, bev_w), dtype=torch.float32, device=device)

    # Obter indices dos pontos validos
    valid = radar_mask > 0.5
    if valid.sum() == 0:
        return grid

    # Extrair pontos validos: [10, N_valid]
    pts = radar_points[:, valid]

    x = pts[0]    # profundidade (frente do carro)
    y = pts[1]    # lateral (esquerda positiva)
    rcs = pts[3]  # radar cross section
    vx = pts[4]   # velocidade x
    vy = pts[5]   # velocidade y

    # Converter coordenadas do mundo para indices do grid
    # nuScenes: x = frente, y = esquerda
    # Grid: row = eixo X (frente), col = eixo Y (esquerda)
    col = ((y - bev_min) / (bev_max - bev_min) * bev_w).long()
    row = ((x - bev_min) / (bev_max - bev_min) * bev_h).long()

    # Filtrar pontos dentro dos limites do grid
    in_bounds = (row >= 0) & (row < bev_h) & (col >= 0) & (col < bev_w)
    row = row[in_bounds]
    col = col[in_bounds]
    rcs_valid = rcs[in_bounds]
    vx_valid = vx[in_bounds]
    vy_valid = vy[in_bounds]

    if row.numel() == 0:
        return grid

    # ── Canal 0: Densidade ──────────────────────────────────────────────
    # Contar quantos pontos caem em cada celula
    indices = row * bev_w + col
    density = torch.zeros(bev_h * bev_w, dtype=torch.float32, device=device)
    density.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32, device=device))
    grid[0] = density.view(bev_h, bev_w)

    # ── Canal 1: RCS medio ──────────────────────────────────────────────
    rcs_sum = torch.zeros(bev_h * bev_w, dtype=torch.float32, device=device)
    rcs_sum.scatter_add_(0, indices, rcs_valid)
    rcs_count = grid[0].view(-1).clone()
    rcs_count[rcs_count == 0] = 1.0  # evitar divisao por zero
    grid[1] = (rcs_sum / rcs_count).view(bev_h, bev_w)

    # ── Canal 2: Velocidade X media ────────────────────────────────────
    vx_sum = torch.zeros(bev_h * bev_w, dtype=torch.float32, device=device)
    vx_sum.scatter_add_(0, indices, vx_valid)
    grid[2] = (vx_sum / rcs_count).view(bev_h, bev_w)

    # ── Canal 3: Velocidade Y media ────────────────────────────────────
    vy_sum = torch.zeros(bev_h * bev_w, dtype=torch.float32, device=device)
    vy_sum.scatter_add_(0, indices, vy_valid)
    grid[3] = (vy_sum / rcs_count).view(bev_h, bev_w)

    return grid


def radar_to_bev_grid_numpy(
    radar_points: np.ndarray,
    radar_mask: np.ndarray,
    bev_size: tuple = (128, 128),
    bev_range: tuple = (-50.0, 50.0),
) -> np.ndarray:
    """
    Versao NumPy (para debug/visualizacao sem GPU).
    Mesma logica de radar_to_bev_grid mas retorna np.ndarray [4, H, W].
    """
    bev_h, bev_w = bev_size
    bev_min, bev_max = bev_range

    grid = np.zeros((4, bev_h, bev_w), dtype=np.float32)

    valid = radar_mask > 0.5
    if valid.sum() == 0:
        return grid

    pts = radar_points[:, valid]

    x = pts[0]
    y = pts[1]
    rcs = pts[3]
    vx = pts[4]
    vy = pts[5]

    col = ((y - bev_min) / (bev_max - bev_min) * bev_w).astype(np.int32)
    row = ((x - bev_min) / (bev_max - bev_min) * bev_h).astype(np.int32)

    in_bounds = (row >= 0) & (row < bev_h) & (col >= 0) & (col < bev_w)
    row = row[in_bounds]
    col = col[in_bounds]
    rcs_valid = rcs[in_bounds]
    vx_valid = vx[in_bounds]
    vy_valid = vy[in_bounds]

    if len(row) == 0:
        return grid

    for i in range(len(row)):
        r, c = row[i], col[i]
        grid[0, r, c] += 1.0
        grid[1, r, c] += rcs_valid[i]
        grid[2, r, c] += vx_valid[i]
        grid[3, r, c] += vy_valid[i]

    # Normalizar RCS e velocidades pela densidade
    density = grid[0].copy()
    density[density == 0] = 1.0
    grid[1] /= density
    grid[2] /= density
    grid[3] /= density

    return grid


def normalize_bev_grid(grid: torch.Tensor) -> torch.Tensor:
    """
    Normaliza cada canal do grid BEV para [-1, 1] usando min-max.
    Args:
        grid: Tensor[C, H, W]
    Returns:
        Tensor[C, H, W] normalizado
    """
    c, h, w = grid.shape
    normalized = torch.zeros_like(grid)

    for i in range(c):
        channel = grid[i]
        c_min = channel.min()
        c_max = channel.max()
        if c_max - c_min > 1e-6:
            normalized[i] = 2.0 * (channel - c_min) / (c_max - c_min) - 1.0
        # Se canal constante, fica zeros

    return normalized
