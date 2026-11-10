"""
Script para visualização e avaliação qualitativa local de predições Camera-Radar BEV Fusion.

Uso:
    python visualize.py --samples 0 5 10 15 20 --threshold 0.4 --save-dir docs/plots
    python visualize.py --num-samples 4 --device cpu
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch

# Garante que 'src' esteja no PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.train import load_config, build_model, load_checkpoint
from src.dataset.nuscenes_dataset import NuScenesFusionDataset
from src.dataset.radar_transforms import radar_to_bev_grid
from src.utils.visualizer import (
    extract_sample_inference,
    draw_bev_fusion_overlay,
    plot_temporal_sequence,
    plot_spatiotemporal_trajectory,
)



def parse_args():
    parser = argparse.ArgumentParser(description="Visualizador Local do BEVFusionDetector")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Caminho do arquivo de configuração",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Caminho para o checkpoint .pth treinado",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="data/sets/nuscenes",
        help="Caminho raiz do dataset nuScenes",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15],
        help="Lista de índices das amostras de validação para plotar (ex: --samples 0 5 10)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Quantidade de amostras sequenciais para plotar a partir do índice 0 (se definido, sobrepõe --samples)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Limiar de probabilidade para binarização da predição (padrão: 0.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Dispositivo de execução ('cuda' ou 'cpu'). Padrão: auto-detect",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="docs/plots",
        help="Diretório onde os gráficos serão salvos em formato PNG",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Se ativado, executa e plota uma sequência temporal consecutiva de frames e sua trajetória",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Índice inicial da amostra para sequência temporal (padrão: 0)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Quantidade de frames consecutivos para a sequência temporal (padrão: 10)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Se ativado, não abre a janela gráfica do matplotlib (útil para servidores headless)",
    )
    parser.add_argument(
        "--ego-color",
        type=str,
        default="#0284c7",
        help="Cor do Ego Vehicle no plot (hex ou nome, ex: '#0284c7', '#22c55e', '#ef4444', '#f59e0b'). Padrão: '#0284c7'",
    )
    parser.add_argument(
        "--ego-sprite",
        type=str,
        default=None,
        help="Caminho opcional para arquivo de imagem PNG com sprite do carro visto de cima (top-down)",
    )
    parser.add_argument(
        "--hide-fov",
        action="store_true",
        help="Se ativado, oculta o cone do campo de visão da câmera frontal no BEV",
    )
    return parser.parse_args()




def process_sample(model, dataset, sample_idx, device, cfg, threshold=0.4):
    sample = dataset[sample_idx]

    image = sample["image"].unsqueeze(0).to(device)
    radar_points = sample["radar_points"].to(device)
    radar_mask = sample["radar_mask"].to(device)

    bev_size = tuple(cfg["data"]["bev_size"])
    bev_range = tuple(cfg["data"]["bev_range"])

    radar_bev = radar_to_bev_grid(
        radar_points,
        radar_mask,
        bev_size=bev_size,
        bev_range=bev_range,
    ).unsqueeze(0)

    with torch.no_grad():
        seg_logits = model(image, radar_bev)
        seg_prob = torch.sigmoid(seg_logits).cpu().squeeze().numpy()

    # Desnormalizar imagem RGB
    img_show = sample["image"].permute(1, 2, 0).cpu().numpy()
    img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_show = np.clip(img_show, 0, 1)

    gt = sample["bev_segmentation"].squeeze().cpu().numpy()
    pred_binary = (seg_prob >= threshold).astype(np.float32)

    # Métricas locais
    intersection = np.logical_and(pred_binary == 1, gt == 1).sum()
    union = np.logical_or(pred_binary == 1, gt == 1).sum()
    iou = float(intersection / (union + 1e-7))

    pred_positives = float(pred_binary.sum())
    gt_positives = float(gt.sum())
    precision = float(intersection / (pred_positives + 1e-7)) if pred_positives > 0 else 0.0
    recall = float(intersection / (gt_positives + 1e-7)) if gt_positives > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall + 1e-7))

    # Mapa de erro RGB
    bev_h, bev_w = gt.shape
    error_map = np.zeros((bev_h, bev_w, 3), dtype=np.float32)
    error_map[..., 1] = (pred_binary == 1) & (gt == 1)  # Verde: True Positive
    error_map[..., 0] = (pred_binary == 1) & (gt == 0)  # Vermelho: False Positive
    error_map[..., 2] = (pred_binary == 0) & (gt == 1)  # Azul: False Negative

    return {
        "sample_idx": sample_idx,
        "img_show": img_show,
        "radar_bev": radar_bev.cpu().squeeze().numpy(),
        "gt": gt,
        "prob": seg_prob,
        "pred_binary": pred_binary,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_map": error_map,
        "num_objects": sample.get("num_objects", 0),
    }


def draw_ego_vehicle(ax, color="#0284c7", sprite_path=None, show_fov=True):
    """
    Desenha o veículo Ego (carro central/próprio) na origem (0, 0) do BEV:
    - Se sprite_path for fornecido e existir como arquivo de imagem (PNG/JPG),
      projeta o sprite do carro visto de cima com as proporções reais.
    - Caso contrário, renderiza uma silhueta vetorial estilizada do carro (chassi,
      cabine, para-brisa frontal, faróis e seta de direção indicando a frente +X).
    - Opcionalmente exibe o cone translúcido do campo de visão (FOV ~70°) da câmera frontal.
    """
    # 1. Cone do Campo de Visão da Câmera Frontal (FOV ~70° = +/- 35° até 50m)
    if show_fov:
        fov_angle = np.deg2rad(35)
        fov_range = 50.0
        # X é frente (vertical), Y é lateral (horizontal)
        fov_poly = Polygon(
            [
                [0.0, 1.5],  # Câmera montada perto do para-brisa (~1.5m à frente do eixo traseiro)
                [fov_range * np.sin(fov_angle), fov_range * np.cos(fov_angle)],
                [-fov_range * np.sin(fov_angle), fov_range * np.cos(fov_angle)],
            ],
            closed=True,
            facecolor="#38bdf8",
            alpha=0.08,
            edgecolor="#0284c7",
            linestyle="--",
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(fov_poly)

    # Dimensões do veículo Ego (Renault Zoe nuScenes: comp ~4.08m, larg ~1.85m)
    # Origem (0, 0) = centro do eixo traseiro.
    # Para-choque traseiro ≈ -0.8m, frente ≈ +3.28m
    car_len = 4.08
    car_wid = 1.85
    y_min, y_max = -car_wid / 2.0, car_wid / 2.0
    x_min, x_max = -0.8, -0.8 + car_len

    # 2. Se fornecido sprite PNG customizado
    if sprite_path and os.path.isfile(sprite_path):
        try:
            sprite_img = plt.imread(sprite_path)
            ax.imshow(
                sprite_img,
                extent=[y_min, y_max, x_min, x_max],
                zorder=10,
                interpolation="bilinear",
            )
            return
        except Exception as e:
            print(f"  [AVISO] Não foi possível carregar o sprite '{sprite_path}': {e}. Usando desenho vetorial.")

    # 3. Desenho Vetorial Tecnológico do Ego Vehicle
    # Chassi principal (cantos arredondados)
    chassis = FancyBboxPatch(
        (y_min, x_min),
        car_wid,
        car_len,
        boxstyle="round,pad=0.08,rounding_size=0.35",
        facecolor=color,
        edgecolor="#ffffff",
        linewidth=1.2,
        alpha=0.95,
        zorder=8,
    )
    ax.add_patch(chassis)

    # Cabine / Teto escuro
    roof_wid = car_wid * 0.72
    roof_len = car_len * 0.50
    roof_ymin = -roof_wid / 2.0
    roof_xmin = x_min + car_len * 0.22
    roof_box = FancyBboxPatch(
        (roof_ymin, roof_xmin),
        roof_wid,
        roof_len,
        boxstyle="round,pad=0.04,rounding_size=0.2",
        facecolor="#0f172a",
        edgecolor="#334155",
        linewidth=0.8,
        alpha=0.92,
        zorder=9,
    )
    ax.add_patch(roof_box)

    # Para-brisa frontal (vidro azulado)
    ws_poly = Polygon(
        [
            [-roof_wid * 0.45, roof_xmin + roof_len],
            [roof_wid * 0.45, roof_xmin + roof_len],
            [roof_wid * 0.38, roof_xmin + roof_len - 0.45],
            [-roof_wid * 0.38, roof_xmin + roof_len - 0.45],
        ],
        closed=True,
        facecolor="#93c5fd",
        alpha=0.95,
        zorder=10,
    )
    ax.add_patch(ws_poly)

    # Faróis dianteiros (LED amarelo/branco)
    hl_left = Rectangle((y_min + 0.15, x_max - 0.25), 0.35, 0.22, facecolor="#fef08a", edgecolor="#ca8a04", linewidth=0.5, zorder=10)
    hl_right = Rectangle((y_max - 0.50, x_max - 0.25), 0.35, 0.22, facecolor="#fef08a", edgecolor="#ca8a04", linewidth=0.5, zorder=10)
    ax.add_patch(hl_left)
    ax.add_patch(hl_right)

    # Seta direcional para a frente (+X)
    ax.arrow(0, x_max + 0.3, 0, 1.8, head_width=1.0, head_length=1.2, fc="#38bdf8", ec="#ffffff", linewidth=0.8, zorder=11)


def plot_single_sample(res, threshold, save_path=None, show=True, ego_color="#0284c7", ego_sprite=None, show_fov=True):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.18)

    # 1. Câmera RGB
    axes[0, 0].imshow(res["img_show"])
    axes[0, 0].set_title(
        f"(a) Câmera Frontal (Amostra #{res['sample_idx']})\nObjetos anotados: {res['num_objects']}",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].axis("off")

    # 2. Radar BEV: Densidade
    radar_density = res["radar_bev"][0]
    im1 = axes[0, 1].imshow(
        radar_density, cmap="viridis", origin="lower", extent=[-50, 50, -50, 50]
    )
    draw_ego_vehicle(axes[0, 1], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
    axes[0, 1].set_title("(b) Radar BEV — Densidade de Pontos", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Y lateral (m) [← Esq | Dir →]")
    axes[0, 1].set_ylabel("X frontal (m)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Radar BEV: Magnitude da Velocidade Doppler
    vx = res["radar_bev"][2]
    vy = res["radar_bev"][3]
    speed = np.sqrt(vx**2 + vy**2)
    im2 = axes[0, 2].imshow(
        speed, cmap="plasma", origin="lower", extent=[-50, 50, -50, 50]
    )
    draw_ego_vehicle(axes[0, 2], color=ego_color, sprite_path=ego_sprite, show_fov=False)
    axes[0, 2].set_title("(c) Radar BEV — Velocidade Doppler (|V| m/s)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Y lateral (m) [← Esq | Dir →]")
    axes[0, 2].set_ylabel("X frontal (m)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Ground Truth BEV
    axes[1, 0].imshow(
        res["gt"], cmap="magma", vmin=0, vmax=1, origin="lower", extent=[-50, 50, -50, 50]
    )
    draw_ego_vehicle(axes[1, 0], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
    axes[1, 0].set_title("(d) Ground Truth BEV (Ocupação Real)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Y lateral (m) [← Esq | Dir →]")
    axes[1, 0].set_ylabel("X frontal (m)")

    # 5. Predição Contínua (Probabilidade Sigmoid)
    im3 = axes[1, 1].imshow(
        res["prob"], cmap="magma", vmin=0, vmax=1, origin="lower", extent=[-50, 50, -50, 50]
    )
    draw_ego_vehicle(axes[1, 1], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
    axes[1, 1].set_title(
        f"(e) Predição da Rede (Probabilidade Sigmoid)\nConfiança contínua [0.0 - 1.0]",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Y lateral (m) [← Esq | Dir →]")
    axes[1, 1].set_ylabel("X frontal (m)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. Diagnóstico de Erros (Overlay TP/FP/FN)
    axes[1, 2].imshow(res["error_map"], origin="lower", extent=[-50, 50, -50, 50])
    draw_ego_vehicle(axes[1, 2], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
    axes[1, 2].set_title(
        f"(f) Diagnóstico de Erro (Limiar {threshold})\nIoU: {res['iou']:.3f} | Prec: {res['precision']:.2f} | Rec: {res['recall']:.2f}\nVerde=TP, Vermelho=FP, Azul=FN",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("Y lateral (m) [← Esq | Dir →]")
    axes[1, 2].set_ylabel("X frontal (m)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"  [SALVO] Amostra #{res['sample_idx']} -> {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_batch_gallery(results, threshold, save_path=None, show=True, ego_color="#0284c7", ego_sprite=None, show_fov=True):
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.8 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, res in enumerate(results):
        # 1. Câmera
        axes[row, 0].imshow(res["img_show"])
        axes[row, 0].set_title(f"Amostra #{res['sample_idx']} — Câmera RGB", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # 2. Radar Densidade
        axes[row, 1].imshow(res["radar_bev"][0], cmap="viridis", origin="lower", extent=[-50, 50, -50, 50])
        draw_ego_vehicle(axes[row, 1], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
        axes[row, 1].set_title("Radar BEV (Densidade)", fontsize=10, fontweight="bold")
        axes[row, 1].set_xlabel("Y lateral (m) [← Esq | Dir →]")
        axes[row, 1].set_ylabel("X frontal (m)")

        # 3. Ground Truth
        axes[row, 2].imshow(res["gt"], cmap="magma", vmin=0, vmax=1, origin="lower", extent=[-50, 50, -50, 50])
        draw_ego_vehicle(axes[row, 2], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
        axes[row, 2].set_title(f"Ground Truth ({res['num_objects']} objetos)", fontsize=10, fontweight="bold")
        axes[row, 2].set_xlabel("Y lateral (m) [← Esq | Dir →]")
        axes[row, 2].set_ylabel("X frontal (m)")

        # 4. Predição Binarizada
        axes[row, 3].imshow(res["pred_binary"], cmap="magma", vmin=0, vmax=1, origin="lower", extent=[-50, 50, -50, 50])
        draw_ego_vehicle(axes[row, 3], color=ego_color, sprite_path=ego_sprite, show_fov=show_fov)
        axes[row, 3].set_title(
            f"Predição (Limiar {threshold}) — IoU: {res['iou']:.3f} | F1: {res['f1']:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[row, 3].set_xlabel("Y lateral (m) [← Esq | Dir →]")
        axes[row, 3].set_ylabel("X frontal (m)")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"  [SALVO] Galeria comparativa -> {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    # 1. Seleciona o dispositivo
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("   Visualizador Local — Camera-Radar BEV Fusion")
    print("=" * 65)
    print(f"Dispositivo selecionado : {device}")
    print(f"Arquivo de Configuração: {args.config}")
    print(f"Checkpoint Treinado    : {args.checkpoint}")
    print(f"Dataroot nuScenes      : {args.dataroot}")
    print(f"Limiar de Binarização  : {args.threshold}")
    print("=" * 65)

    # 2. Carrega Configuração
    cfg = load_config(args.config)
    cfg["data"]["dataroot"] = args.dataroot

    # 3. Constrói Modelo e Carrega Pesos
    print("\n1. Instanciando modelo BEVFusionDetector...")
    model = build_model(cfg, device)

    print(f"2. Carregando pesos de {args.checkpoint}...")
    epoch, metrics = load_checkpoint(model, optimizer=None, path=args.checkpoint, device=device)
    model.eval()
    print(f"✔ Checkpoint restaurado da época {epoch}!")
    if metrics:
        print(f"  Métricas registradas: {metrics}")

    # 4. Carrega Dataset
    print(f"\n3. Conectando ao nuScenes v1.0-mini em {args.dataroot}...")
    val_dataset = NuScenesFusionDataset(
        dataroot=args.dataroot,
        version=cfg["data"]["version"],
        split="val",
        train_ratio=cfg["data"]["train_ratio"],
        image_size=tuple(cfg["data"]["image_size"]),
        bev_size=tuple(cfg["data"]["bev_size"]),
        bev_range=tuple(cfg["data"]["bev_range"]),
        radar_max_points=cfg["data"]["radar_max_points"],
    )
    print(f"✔ Conjunto de validação pronto com {len(val_dataset)} amostras.")

    # 5. Define amostras a processar (se não estiver em modo de sequência pura)
    results = []
    if not args.sequence:
        if args.num_samples is not None:
            sample_indices = list(range(min(args.num_samples, len(val_dataset))))
        else:
            sample_indices = [idx for idx in args.samples if idx < len(val_dataset)]

        print(f"\n4. Gerando inferência para {len(sample_indices)} amostras: {sample_indices}...")

        for idx in sample_indices:
            res = process_sample(
                model=model,
                dataset=val_dataset,
                sample_idx=idx,
                device=device,
                cfg=cfg,
                threshold=args.threshold,
            )
            results.append(res)
            print(
                f"   Amostra #{idx:02d} | Objetos: {res['num_objects']:02d} | "
                f"IoU: {res['iou']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1']:.4f}"
            )

            single_save_path = os.path.join(args.save_dir, f"sample_{idx:03d}_detailed.png")
            plot_single_sample(
                res,
                threshold=args.threshold,
                save_path=single_save_path,
                show=(not args.no_show),
                ego_color=args.ego_color,
                ego_sprite=args.ego_sprite,
                show_fov=(not args.hide_fov),
            )


    # 6. Modo Sequência Temporal (Simulação de Movimento Retilíneo)
    if args.sequence:
        seq_start = args.start_idx
        seq_end = min(len(val_dataset), seq_start + args.num_frames)
        print(f"\n5. Gerando Sequência Temporal ({args.num_frames} frames: #{seq_start} a #{seq_end - 1})...")
        
        seq_results = []
        for s_idx in range(seq_start, seq_end):
            s_res = extract_sample_inference(
                model=model,
                dataset=val_dataset,
                sample_idx=s_idx,
                device=device,
                cfg=cfg,
                threshold=args.threshold,
            )
            seq_results.append(s_res)
            print(f"   Frame #{s_idx:02d} | IoU: {s_res['iou']:.3f} | Alvos GT: {s_res['num_objects']}")
            
        seq_save_path = os.path.join(args.save_dir, f"sequence_{seq_start}_to_{seq_end - 1}.png")
        traj_save_path = os.path.join(args.save_dir, f"trajectory_{seq_start}_to_{seq_end - 1}.png")
        
        plot_temporal_sequence(
            seq_results,
            threshold=args.threshold,
            dt=0.5,
            save_path=seq_save_path,
            show=(not args.no_show),
        )
        plot_spatiotemporal_trajectory(
            seq_results,
            threshold=args.threshold,
            dt=0.5,
            save_path=traj_save_path,
            show=(not args.no_show),
        )

    # 7. Salva Galeria Geral (se amostras pontuais foram processadas)
    if len(results) > 1 and not args.sequence:
        gallery_save_path = os.path.join(args.save_dir, "gallery_comparison.png")
        plot_batch_gallery(
            results,
            threshold=args.threshold,
            save_path=gallery_save_path,
            show=(not args.no_show),
            ego_color=args.ego_color,
            ego_sprite=args.ego_sprite,
            show_fov=(not args.hide_fov),
        )

    print("\n" + "=" * 65)
    print(f"✔ Visualizações concluídas com sucesso!")
    print(f"  Diretório dos gráficos salvos: {os.path.abspath(args.save_dir)}")
    print("=" * 65)



if __name__ == "__main__":
    main()
