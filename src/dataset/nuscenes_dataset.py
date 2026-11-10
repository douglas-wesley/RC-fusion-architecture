"""
PyTorch Dataset para nuScenes Camera-Radar Fusion (BEV).

Extraido e adaptado do CRF-Net NuscenesGenerator (TUMFTM/CameraRadarFusionNet).
Reescrito em PyTorch puro, sem dependencias de Keras/TF.

Formato de saida:
    {
        "image":             Tensor[C, H, W]             — RGB normalizado
        "radar_points":      Tensor[n_channels, max_pts]  — radar bruto (padded)
        "radar_mask":        Tensor[max_pts]              — 1=pt real, 0=padding
        "bev_segmentation":  Tensor[1, bev_h, bev_w]     — GT occupancy
        "num_objects":       int
    }
"""

import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility


# ── Classes suportadas (mapeamento nuScenes → id) ──────────────────────────
CATEGORY_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 1,
    "vehicle.bus.bendy": 2,
    "vehicle.bus.rigid": 2,          # bus.bendy e bus.rigid → mesma classe
    "vehicle.bicycle": 3,
    "vehicle.motorcycle": 4,
    "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5,
    "human.pedestrian.construction_worker": 5,
    "human.pedestrian.police_officer": 5,
}

# Canais do radar nuScenes (18 canais brutos)
RADAR_RAW_CHANNELS = [
    "x", "y", "z", "dyn_prop", "id", "rcs",
    "vx", "vy", "vx_comp", "vy_comp", "is_quality_valid",
    "ambig_state", "x_rms", "y_rms", "invalid_state", "pdh0",
    "vx_rms", "vy_rms",
]


class NuScenesFusionDataset(Dataset):
    """Dataset PyTorch para fusao Camera-Radar em BEV com nuScenes."""

    def __init__( # Origem do 
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "train",
        train_ratio: float = 0.7,
        image_size: Tuple[int, int] = (384, 672), 
        bev_size: Tuple[int, int] = (128, 128),
        bev_range: Tuple[float, float] = (-50.0, 50.0),
        radar_max_points: int = 200,
        category_mapping: Optional[Dict[str, str]] = None,
        transform=None,
    ):
        """
        Args:
            dataroot: Caminho raiz do nuScenes (contem samples/, v1.0-mini/, etc.)
            version: Versao do dataset (v1.0-mini, v1.0-trainval)
            split: 'train' ou 'val'
            train_ratio: Fracao de scenes para treino
            image_size: (H, W) da imagem de saida
            bev_size: (H, W) do grid BEV
            bev_range: (min_m, max_m) — range em metros nos eixos X e Y
            radar_max_points: Numero fixo de pontos radar para padding
            category_mapping: Dict customizado de mapeamento de categorias
            transform: Transformacao opcional aplicada a imagem
        """
        super().__init__()
        self.dataroot = dataroot
        self.image_size = image_size
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.radar_max_points = radar_max_points
        self.transform = transform
        self.category_mapping = category_mapping or CATEGORY_MAP

        # Inicializar nuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Separar scenes em train/val
        scenes = self.nusc.scene
        n_train = int(len(scenes) * train_ratio)

        if split == "train":
            selected_scenes = scenes[:n_train]
        else:
            selected_scenes = scenes[n_train:]

        # Coletar todos os sample tokens das scenes selecionadas
        self.sample_tokens: List[str] = []
        for scene in selected_scenes:
            token = scene["first_sample_token"]
            for _ in range(scene["nbr_samples"]):
                self.sample_tokens.append(token)
                sample = self.nusc.get("sample", token)
                token = sample["next"] if sample["next"] else None
                if token is None:
                    break

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)

        # ── Carregar imagem ──────────────────────────────────────────────
        image = self._load_image(sample)

        # ── Carregar radar ───────────────────────────────────────────────
        radar_points, radar_mask = self._load_radar(sample)

        # ── Carregar anotacoes ──────────────────────────────────────────
        bev_gt = self._load_bev_segmentation_gt(sample)
        num_objects = int(bev_gt.sum())

        # ── Aplicar transform na imagem ─────────────────────────────────
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "radar_points": radar_points,
            "radar_mask": radar_mask,
            "bev_segmentation": bev_gt,
            "num_objects": num_objects,
        }

    # =========================================================================
    # Metodos internos
    # =========================================================================

    def _load_image(self, sample: dict) -> torch.Tensor:
        """
        Carrega a imagem da CAM_FRONT, redimensiona e normaliza.
        Retorna: Tensor[3, H, W]
        """
        cam_token = sample["data"]["CAM_FRONT"]
        sd_rec = self.nusc.get("sample_data", cam_token)
        img_path = osp.join(self.nusc.dataroot, sd_rec["filename"])

        # Ler com OpenCV (BGR) e converter para RGB
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Imagem nao encontrada: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Redimensionar
        w, h = self.image_size[1], self.image_size[0]
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalizar para [0, 1] e depois para ImageNet
        img = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC → CHW
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img

    def _load_radar(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carrega o radar da RADAR_FRONT, extrai canais relevantes e faz padding.
        Retorna:
            radar_points: Tensor[n_channels, radar_max_points]
            radar_mask:   Tensor[radar_max_points]  (1=real, 0=padding)
        """
        radar_token = sample["data"]["RADAR_FRONT"]
        sd_rec = self.nusc.get("sample_data", radar_token)
        radar_path = osp.join(self.nusc.dataroot, sd_rec["filename"])

        try:
            pc = RadarPointCloud.from_file(radar_path)
            points = pc.points.astype(np.float32)  # shape: [18, N]
        except Exception:
            # Se radar vazio ou erro de leitura, retorna zeros
            points = np.zeros((18, 0), dtype=np.float32)

        n_channels, n_points = points.shape

        # Filtrar pontos invalidos
        if n_points > 0:
            # invalid_state == 0 significa ponto valido (ou inválido)
            valid_mask = points[14, :] == 0  # invalid_state channel
            points = points[:, valid_mask]
            n_points = points.shape[1]

        # Extrair canais relevantes para BEV:
        # x(0), y(1), z(2), rcs(5), vx(6), vy(7) + 4 derivados
        # Total: 10 canais (x, y, z, rcs, vx, vy, vx_comp, vy_comp, distance, azimuth)
        if n_points > 0:
            # Aplicar calibracao extrinseca: converter do referencial do sensor para o Ego-Vehicle
            cs_rec = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            rot_matrix = Quaternion(cs_rec["rotation"]).rotation_matrix
            trans = np.array(cs_rec["translation"]).reshape(3, 1)
            points[:3, :] = (rot_matrix @ points[:3, :] + trans).astype(np.float32)

            x = points[0:1, :]   # [1, N]
            y = points[1:2, :]
            z = points[2:3, :]
            rcs = points[5:6, :]
            vx = points[6:7, :]
            vy = points[7:8, :]
            vx_comp = points[8:9, :]
            vy_comp = points[9:10, :]

            # Calcular distancia e azimuth
            dist = np.sqrt(x**2 + y**2)
            azimuth = np.arctan2(y, x)

            # Empilhar: [10, N]
            radar_features = np.concatenate(
                [x, y, z, rcs, vx, vy, vx_comp, vy_comp, dist, azimuth], axis=0
            )
        else:
            radar_features = np.zeros((10, 0), dtype=np.float32)

        n_features = radar_features.shape[0]

        # Padding ate radar_max_points
        padded = np.zeros((n_features, self.radar_max_points), dtype=np.float32)
        mask = np.zeros(self.radar_max_points, dtype=np.float32)

        n_copy = min(n_points, self.radar_max_points)
        if n_copy > 0:
            padded[:, :n_copy] = radar_features[:, :n_copy]
            mask[:n_copy] = 1.0

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
        )

    def _load_bev_segmentation_gt(self, sample: dict) -> torch.Tensor:
        """
        Gera o ground truth de segmentacao BEV (mapa de ocupacao).
        Para cada bounding box 3D, converter para o referencial do ego vehicle,
        projetar no grid BEV e desenhar o poligono rotacionado com cv2.fillPoly.
        Retorna: Tensor[1, bev_h, bev_w]
        """
        bev_h, bev_w = self.bev_size
        bev_min, bev_max = self.bev_range
        canvas = np.zeros((bev_h, bev_w), dtype=np.float32)

        # 1. Obter a pose do ego correspondente ao frame da CAM_FRONT
        cam_token = sample["data"]["CAM_FRONT"]
        sd_rec = self.nusc.get("sample_data", cam_token)
        ego_pose = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        # 2. Iterar sobre as anotacoes de objetos da cena
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Filtrar apenas as categorias configuradas em self.category_mapping
            if ann["category_name"] not in self.category_mapping:
                continue

            # Ignorar objetos com status de visibilidade nula/invisivel
            if ann["visibility_token"] == "1":
                continue

            # Instanciar Box 3D no referencial global e converter para o referencial do Ego
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
            box.translate(-np.array(ego_pose["translation"]))
            box.rotate(Quaternion(ego_pose["rotation"]).inverse)

            # 1. Filtrar pelo Campo de Visao Frontal (FOV):
            # A Câmera Frontal e o Radar Frontal cobrem apenas o hemisferio frontal.
            # Descartar objetos traseiros (X <= 0.5m) e fora da abertura angular (> 45 graus)
            cx, cy, _ = box.center
            if cx < 0.5:
                continue
            azimuth_deg = np.rad2deg(np.arctan2(abs(cy), cx))
            if azimuth_deg > 45.0:
                continue

            # 2. Extrair os 4 vertices da base no solo na ordem perimetrica correta (BEVCar)
            # Indices no nuScenes: 4 (Front-Left), 5 (Front-Right), 6 (Rear-Right), 7 (Rear-Left)
            corners_bottom = box.corners()[:2, [4, 5, 6, 7]]
            x_pts = corners_bottom[0]
            # No nuScenes Ego: +Y e Esquerda, -Y e Direita.
            # Invertemos o sinal (-Y) para que no Grid/BEV: Esquerda seja -Y e Direita seja +Y,
            # coincidindo 100% com a imagem da camera frontal (visao do motorista).
            y_pts = -corners_bottom[1]

            # Discretizar coordenadas metricas para indices de linha (X) e coluna (Y) do grid BEV
            rows = np.round(((x_pts - bev_min) / (bev_max - bev_min)) * bev_h).astype(np.int32)
            cols = np.round(((y_pts - bev_min) / (bev_max - bev_min)) * bev_w).astype(np.int32)

            # Desenhar poligono solido com a orientacao (yaw) exata do veiculo
            poly_pts = np.stack([cols, rows], axis=1)
            cv2.fillPoly(canvas, [poly_pts], 1.0)

        grid = np.expand_dims(canvas, axis=0)
        return torch.from_numpy(grid)


    def get_camera_calibration(self, sample: dict) -> Dict[str, np.ndarray]:
        """
        Retorna intrinsecas e extrinsecas da CAM_FRONT.
        Util para o View Transformer (IPM).
        """
        cam_token = sample["data"]["CAM_FRONT"]
        sd_rec = self.nusc.get("sample_data", cam_token)
        cs_rec = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])

        intrinsic = np.array(cs_rec["camera_intrinsic"])  # 3x3
        extrinsic_rot = Quaternion(cs_rec["rotation"]).rotation_matrix  # 3x3
        extrinsic_trans = np.array(cs_rec["translation"])  # (3,)

        # Matriz extrinseca 4x4
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = extrinsic_rot
        extrinsic[:3, 3] = extrinsic_trans

        return {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "translation": extrinsic_trans,
            "rotation": extrinsic_rot,
        }

    def get_class_name(self, label_id: int) -> str:
        """Retorna o nome da classe a partir do ID."""
        inverse_map = {
            0: "vehicle.car",
            1: "vehicle.truck",
            2: "vehicle.bus",
            3: "vehicle.bicycle",
            4: "vehicle.motorcycle",
            5: "human.pedestrian",
        }
        return inverse_map.get(label_id, "unknown")


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Collate function personalizado para lidar com radar de tamanho variavel.
    Faz padding uniforme para todos os itens do batch.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    radar_points = torch.stack([b["radar_points"] for b in batch], dim=0)
    radar_mask = torch.stack([b["radar_mask"] for b in batch], dim=0)
    bev_gt = torch.stack([b["bev_segmentation"] for b in batch], dim=0)
    num_objects = [b["num_objects"] for b in batch]

    return {
        "image": images,
        "radar_points": radar_points,
        "radar_mask": radar_mask,
        "bev_segmentation": bev_gt,
        "num_objects": num_objects,
    }
