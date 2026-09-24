# Camera-Radar BEV Fusion

Pipeline de deteccao e classificacao de objetos por fusao Camera+Radar em espaco BEV (Bird's-Eye View), utilizing o dataset nuScenes.

## Estrutura

```
fusion/
├── config/              # Hiperparametros (default.yaml)
├── src/
│   ├── dataset/         # PyTorch Dataset + radar transforms
│   ├── models/          # Arquitetura da rede
│   │   ├── backbones/   # MobileNetV2 / ResNet18
│   │   └── modules/     # View transformer + Fusion module
│   ├── utils/           # Helpers e visualizacao
│   └── engine/          # Training + evaluation loops
├── tests/               # Sanity checks (rodam em CPU)
├── notebooks/           # Notebooks Colab para treino
└── data/                # nuScenes dataset (nao versionado)
```

## Setup

```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 2. Instalar PyTorch (conforme sua GPU)
# Veja: https://pytorch.org/get-started/locally/

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Baixar nuScenes mini
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz -C data/sets/
```

## Uso

```bash
# Testar pipeline (CPU)
python tests/test_pipeline.py

# Treinar (requer GPU — usar Colab)
python src/engine/train.py --config config/default.yaml --dataroot ./data/sets/nuscenes
```

## Google Colab

1. Sincronizar a pasta `fusion/` com o Google Drive
2. Abrir `notebooks/02_train.ipynb` no Colab
3. Seguir as instrucoes no notebook
