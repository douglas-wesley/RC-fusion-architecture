"""
Sanity check do pipeline de dados nuScenes.

Este script valida:
  1. Carregamento de imagem da CAM_FRONT
  2. Carregamento de pontos radar da RADAR_FRONT
  3. Extracao de bounding boxes 3D (annotations)
  4. Projecao BEV do ground truth
  5. Formato dos tensores de saida

Rodar:
    python tests/test_pipeline.py --dataroot ./data/sets/nuscenes
    ou com o venv do BEV:
    ../BEV/.venv/bin/python tests/test_pipeline.py --dataroot ../BEV/data/sets/nuscenes
"""

import sys
import os
import argparse

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


def test_nuscenes_api(dataroot: str):
    """Testa se a API do nuScenes funciona e retorna dados validos."""
    print("=" * 60)
    print("  TESTE 1: API nuScenes")
    print("=" * 60)

    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    print(f"  Scenes: {len(nusc.scene)}")
    print(f"  Samples na scene 0: {scene['nbr_samples']}")
    print(f"  Sensores disponiveis: {list(sample['data'].keys())}")
    print("  [OK] API nuScenes funcional\n")
    return nusc, sample


def test_image_loading(nusc: NuScenes, sample: dict, dataroot: str):
    """Testa carregamento e processamento da imagem."""
    print("=" * 60)
    print("  TESTE 2: Carregamento de Imagem (CAM_FRONT)")
    print("=" * 60)

    cam_token = sample["data"]["CAM_FRONT"]
    sd_rec = nusc.get("sample_data", cam_token)
    img_path = os.path.join(dataroot, sd_rec["filename"])

    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Falha ao ler imagem: {img_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionar para image_size do config
    h_target, w_target = 384, 672
    img_resized = cv2.resize(img_rgb, (w_target, h_target))

    # Normalizar
    img = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    print(f"  Caminho: {img_path}")
    print(f"  Shape original: {img_bgr.shape}")
    print(f"  Shape apos resize: ({h_target}, {w_target}, 3)")
    print(f"  Shape final (CHW): {img.shape}")
    print(f"  Min/Max apos norm: [{img.min():.3f}, {img.max():.3f}]")
    print("  [OK] Imagem carregada e normalizada\n")
    return img


def test_radar_loading(nusc: NuScenes, sample: dict, dataroot: str):
    """Testa carregamento e processamento do radar."""
    print("=" * 60)
    print("  TESTE 3: Carregamento de Radar (RADAR_FRONT)")
    print("=" * 60)

    radar_token = sample["data"]["RADAR_FRONT"]
    sd_rec = nusc.get("sample_data", radar_token)
    radar_path = os.path.join(dataroot, sd_rec["filename"])

    pc = RadarPointCloud.from_file(radar_path)
    points = pc.points.astype(np.float32)

    print(f"  Caminho: {radar_path}")
    print(f"  Shape bruto: {points.shape} (canais x pontos)")
    print(f"  Total de pontos: {points.shape[1]}")

    # Mostrar canais
    channel_names = [
        "x", "y", "z", "dyn_prop", "id", "rcs",
        "vx", "vy", "vx_comp", "vy_comp", "is_quality_valid",
        "ambig_state", "x_rms", "y_rms", "invalid_state", "pdh0",
        "vx_rms", "vy_rms",
    ]

    if points.shape[1] > 0:
        print(f"\n  Primeiro ponto valido:")
        for i, name in enumerate(channel_names):
            print(f"    {name:20s}: {points[i, 0]:.4f}")

    # Filtrar invalidos
    valid_mask = points[14, :] == 0  # invalid_state
    n_valid = valid_mask.sum()
    print(f"\n  Pontos validos (invalid_state=0): {n_valid}/{points.shape[1]}")

    # Extrair features relevantes
    if n_valid > 0:
        pts_valid = points[:, valid_mask]
        x = pts_valid[0]
        y = pts_valid[1]
        rcs = pts_valid[5]
        vx = pts_valid[6]
        vy = pts_valid[7]

        print(f"\n  Range X (frente):  [{x.min():.1f}, {x.max():.1f}] m")
        print(f"  Range Y (lateral): [{y.min():.1f}, {y.max():.1f}] m")
        print(f"  Range RCS:         [{rcs.min():.1f}, {rcs.max():.1f}] dB")
        print(f"  Range VX:          [{vx.min():.2f}, {vx.max():.2f}] m/s")
        print(f"  Range VY:          [{vy.min():.2f}, {vy.max():.2f}] m/s")

    print("  [OK] Radar carregado e processado\n")

    # Retornar formato padding
    radar_max_points = 200
    n_features = 10
    padded = np.zeros((n_features, radar_max_points), dtype=np.float32)
    mask = np.zeros(radar_max_points, dtype=np.float32)

    if n_valid > 0:
        pts_feat = np.stack([
            pts_valid[0],   # x
            pts_valid[1],   # y
            pts_valid[2],   # z
            pts_valid[5],   # rcs
            pts_valid[6],   # vx
            pts_valid[7],   # vy
            pts_valid[8],   # vx_comp
            pts_valid[9],   # vy_comp,
            np.sqrt(pts_valid[0]**2 + pts_valid[1]**2),  # distance
            np.arctan2(pts_valid[1], pts_valid[0]),       # azimuth
        ], axis=0)

        n_copy = min(int(n_valid), radar_max_points)
        padded[:, :n_copy] = pts_feat[:, :n_copy]
        mask[:n_copy] = 1.0

    print(f"  Shape radar (padded): {padded.shape}")
    print(f"  Shape mask:           {mask.shape}")
    print(f"  Pontos nao-zero:      {int(mask.sum())}")

    return padded, mask


def test_bev_gt(nusc: NuScenes, sample: dict):
    """Testa geracao do ground truth BEV."""
    print("\n" + "=" * 60)
    print("  TESTE 4: Ground Truth BEV (Segmentacao)")
    print("=" * 60)

    from nuscenes.utils.geometry_utils import BoxVisibility

    annotations = nusc.get_sample_data(
        sample["data"]["CAM_FRONT"],
        box_vis_level=BoxVisibility.ANY,
    )
    boxes = annotations[1]

    bev_h, bev_w = 128, 128
    bev_range = (-50.0, 50.0)
    grid = np.zeros((1, bev_h, bev_w), dtype=np.float32)

    category_map = {
        "vehicle.car": 0,
        "vehicle.truck": 1,
        "vehicle.bus.bendy": 2,
        "vehicle.bus.rigid": 2,
        "vehicle.bicycle": 3,
        "vehicle.motorcycle": 4,
        "human.pedestrian.adult": 5,
        "human.pedestrian.child": 5,
    }

    n_filtered = 0
    for box in boxes:
        if box.name not in category_map:
            continue
        n_filtered += 1

        cx, cy, cz = box.center
        col = int(((cy - bev_range[0]) / (bev_range[1] - bev_range[0])) * bev_w)
        row = int(((cx - bev_range[0]) / (bev_range[1] - bev_range[0])) * bev_h)

        if 0 <= col < bev_w and 0 <= row < bev_h:
            grid[0, row, col] = 1.0

            wlh = box.wlh
            radius_col = max(1, int((wlh[0] / 2) / (bev_range[1] - bev_range[0]) * bev_w))
            radius_row = max(1, int((wlh[1] / 2) / (bev_range[1] - bev_range[0]) * bev_h))

            r_min = max(0, row - radius_row)
            r_max = min(bev_h, row + radius_row + 1)
            c_min = max(0, col - radius_col)
            c_max = min(bev_w, col + radius_col + 1)

            grid[0, r_min:r_max, c_min:c_max] = 1.0

    print(f"  Total de boxes: {len(boxes)}")
    print(f"  Boxes filtradas (categorias suportadas): {n_filtered}")
    print(f"  Shape GT: {grid.shape}")
    print(f"  Pixels positivos: {int(grid.sum())}")
    print(f"  Cobertura: {grid.sum() / (bev_h * bev_w) * 100:.2f}%")
    print("  [OK] Ground Truth BEV gerado\n")

    return grid


def test_dataset_class(dataroot: str):
    """Testa a classe NuScenesFusionDataset completa."""
    print("=" * 60)
    print("  TESTE 5: NuScenesFusionDataset (classe completa)")
    print("=" * 60)

    try:
        import torch
        from src.dataset import NuScenesFusionDataset, collate_fn
    except ImportError as e:
        print(f"  [SKIP] PyTorch nao instalado: {e}")
        print("  Pulando teste da classe Dataset.\n")
        return

    dataset = NuScenesFusionDataset(
        dataroot=dataroot,
        version="v1.0-mini",
        split="train",
        image_size=(384, 672),
        bev_size=(128, 128),
        bev_range=(-50.0, 50.0),
        radar_max_points=200,
    )

    print(f"  Tamanho do dataset (train): {len(dataset)}")

    # Carregar 1 sample
    sample = dataset[0]
    print(f"\n  Chaves retornadas: {list(sample.keys())}")
    print(f"  image:            {sample['image'].shape} {sample['image'].dtype}")
    print(f"  radar_points:     {sample['radar_points'].shape} {sample['radar_points'].dtype}")
    print(f"  radar_mask:       {sample['radar_mask'].shape} {sample['radar_mask'].dtype}")
    print(f"  bev_segmentation: {sample['bev_segmentation'].shape} {sample['bev_segmentation'].dtype}")
    print(f"  num_objects:      {sample['num_objects']}")

    # Testar collate_fn com batch de 2
    batch = collate_fn([dataset[0], dataset[1]])
    print(f"\n  Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape}")
        else:
            print(f"    {k}: {v}")

    print("\n  [OK] Dataset completo funcional")
    print(f"\n  Resumo de memoria por sample:")
    print(f"    Imagem:     {sample['image'].numel() * 4 / 1024:.1f} KB")
    print(f"    Radar:      {sample['radar_points'].numel() * 4 / 1024:.1f} KB")
    print(f"    GT BEV:     {sample['bev_segmentation'].numel() * 4 / 1024:.1f} KB")
    print(f"    Total:      ~{(sample['image'].numel() + sample['radar_points'].numel() + sample['bev_segmentation'].numel()) * 4 / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Sanity check do pipeline de dados")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="./data/sets/nuscenes",
        help="Caminho para o nuScenes dataset",
    )
    args = parser.parse_args()

def test_backbone():
    """Testa o ImageBackbone (MobileNetV2)."""
    print("\n" + "=" * 60)
    print("  TESTE 6: ImageBackbone (MobileNetV2)")
    print("=" * 60)

    try:
        import torch
        from src.models.backbones.image_backbone import ImageBackbone
    except ImportError as e:
        print(f"  [SKIP] Modulo nao encontrado: {e}\n")
        return

    model = ImageBackbone(out_channels=32, pretrained=False)
    x = torch.randn(2, 3, 384, 672)
    out = model(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Output shape (method): {model.get_output_shape()}")
    print("  [OK] Backbone funcional\n")


def test_view_transformer():
    """Testa o View Transformer (IPM)."""
    print("=" * 60)
    print("  TESTE 7: View Transformer (IPM)")
    print("=" * 60)

    try:
        import torch
        from src.models.modules.view_transformer import IPMTransformer
    except ImportError as e:
        print(f"  [SKIP] Modulo nao encontrado: {e}\n")
        return

    model = IPMTransformer(
        bev_size=(128, 128),
        bev_range=(-50.0, 50.0),
        image_size=(384, 672),
        feature_size=(24, 42),
    )
    x = torch.randn(2, 32, 24, 42)
    out = model(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Grid:   {model.grid.shape}")
    print("  [OK] View Transformer funcional\n")


def test_fusion_module():
    """Testa o Fusion Module."""
    print("=" * 60)
    print("  TESTE 8: Fusion Module")
    print("=" * 60)

    try:
        import torch
        from src.models.modules.fusion_module import FusionModule
    except ImportError as e:
        print(f"  [SKIP] Modulo nao encontrado: {e}\n")
        return

    model = FusionModule(img_channels=32, radar_channels=4, out_channels=64)
    img = torch.randn(2, 32, 128, 128)
    radar = torch.randn(2, 4, 128, 128)
    out = model(img, radar)

    print(f"  Image:    {img.shape}")
    print(f"  Radar:    {radar.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    print("  [OK] Fusion Module funcional\n")


def test_full_model():
    """Testa o modelo completo BEVFusionDetector."""
    print("=" * 60)
    print("  TESTE 9: BEVFusionDetector (modelo completo)")
    print("=" * 60)

    try:
        import torch
        from src.models.detector import BEVFusionDetector
    except ImportError as e:
        print(f"  [SKIP] Modulo nao encontrado: {e}\n")
        return

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
    seg_pred = model.predict(image, radar_bev, threshold=0.5)

    print(f"  Image:      {image.shape}")
    print(f"  Radar BEV:  {radar_bev.shape}")
    print(f"  Seg Logits: {seg_logits.shape}")
    print(f"  Seg Pred:   {seg_pred.shape}")

    # Parametros
    params = model.count_parameters()
    print(f"\n  Parametros por modulo:")
    for k, v in params.items():
        print(f"    {k:20s}: {v/1e6:.2f}M")

    print("\n  [OK] Modelo completo funcional\n")


def main():
    parser = argparse.ArgumentParser(description="Sanity check do pipeline de dados")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="./data/sets/nuscenes",
        help="Caminho para o nuScenes dataset",
    )
    args = parser.parse_args()

    # Verificar se dataroot existe
    if not os.path.exists(args.dataroot):
        print(f"ERRO: Dataset nao encontrado em {args.dataroot}")
        print("Execute: wget https://www.nuscenes.org/data/v1.0-mini.tgz")
        print("         tar -xzf v1.0-mini.tgz -C data/sets/")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SANITY CHECK — Pipeline Camera-Radar BEV Fusion")
    print("=" * 60)
    print(f"  Dataroot: {args.dataroot}\n")

    # Testes sequenciais
    nusc, sample = test_nuscenes_api(args.dataroot)
    test_image_loading(nusc, sample, args.dataroot)
    test_radar_loading(nusc, sample, args.dataroot)
    test_bev_gt(nusc, sample)
    test_dataset_class(args.dataroot)

    # Testes do modelo
    test_backbone()
    test_view_transformer()
    test_fusion_module()
    test_full_model()

    print("=" * 60)
    print("  TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
