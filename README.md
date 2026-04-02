# MRI Motion Artifact Classification

Investigating how synthetic motion artifacts in brain MRI scans affect the accuracy of Alzheimer's Disease (AD) classification using a Vision Transformer (ViT).

## Overview

This project generates synthetic motion artifacts on OASIS brain MRI slices at five severity levels (M0–M4) using Cartesian k-space sampling, then measures the degradation of a ViT-based AD classifier trained on clean (M0) images.

### Key Findings

| Motion Level | Accuracy | Precision | Recall | F1 Score | AUROC |
|---|---|---|---|---|---|
| M0 (No Motion) | 0.4452 | 0.6811 | 0.4200 | 0.5196 | 0.4658 |
| M1 (Small) | 0.3429 | 0.7308 | 0.1267 | 0.2159 | 0.3508 |
| M2 (Mild) | 0.4048 | 0.8472 | 0.2033 | 0.3280 | 0.3672 |
| M3 (Moderate) | 0.3310 | 0.6939 | 0.1133 | 0.1948 | 0.3667 |
| M4 (Severe) | 0.3429 | 0.7609 | 0.1167 | 0.2023 | 0.4191 |

## Architecture

- **Vision Transformer (ViT)**: Custom implementation with 6 transformer blocks, 384-dim embeddings, 16×16 patches on 256×256 grayscale MRI slices
- **Motion Simulation**: Rigid-body rotations applied in k-space using Cartesian sampling to produce realistic slice-dependent motion artifacts
- **AD Labels**: Binary classification (healthy vs. dementia) derived from Clinical Dementia Rating (CDR) scores in the [OASIS](https://www.oasis-brains.org/) cross-sectional dataset

## Project Structure

```
├── MainADExperiment.py          # Train on clean MRI, evaluate across motion levels
├── MainViT.py                   # ViT training for motion severity classification
├── Main.py                      # Legacy Keras CNN training
├── DeepLearning/
│   ├── ViTModel.py              # Vision Transformer (PyTorch)
│   ├── CNNModel.py              # CNN model (Keras)
│   ├── VAENetwork.py            # Variational Autoencoder
│   └── CycleGAN.py              # CycleGAN for domain adaptation
├── Utils/
│   ├── DataUtils/
│   │   ├── ADDataLoader.py      # OASIS AD dataset with CDR labels
│   │   ├── DataGenerator.py     # Data generation pipeline
│   │   ├── PyTorchDataLoader.py # PyTorch dataset/dataloader
│   │   └── DataLoader.py        # Legacy data loading
│   ├── MotionUtils/
│   │   ├── GenerateMotion.py    # Synthetic motion artifact generation
│   │   └── ImageTransform.py    # Image transformation utilities
│   └── kspace/
│       └── CartesianSampler.py  # K-space Cartesian sampling
├── tables.py                    # Parse results into publication-ready tables
├── data/                        # Exported CSV/Excel result tables
└── tests/                       # Unit tests
```

## Setup

```bash
# Clone the repository
git clone https://github.com/NishantKC/MRI_Motion_Classification.git
cd MRI_Motion_Classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download the [OASIS cross-sectional dataset](https://www.oasis-brains.org/)
2. Place NIfTI files in the expected directory and update paths in `Utils/DataUtils/DataGenerator.py`
3. Generate motion artifacts:
   ```bash
   python -m Utils.MotionUtils.GenerateMotion
   ```

## Usage

### AD Classification Experiment

Train on clean images (M0) and evaluate across all motion levels:

```bash
python MainADExperiment.py
```

This produces:
- `ad_motion_metrics_results.csv` — metrics per motion level
- `ad_motion_all_metrics.png` — individual metric bar charts
- `ad_motion_combined_metrics.png` — grouped bar chart

### Motion Severity Classification (ViT)

```bash
python MainViT.py
```

### Generate Publication Tables

```bash
python tables.py
```

Outputs CSV and Excel files in `data/`.

## Tests

```bash
python -m pytest tests/
```

## Reference

Mohebbian, M., Walia, E., Habibullah, M., Stapleton, S. and Wahid, K.A., 2021. Classifying MRI motion severity using a stacked ensemble approach. *Magnetic Resonance Imaging*, 75, pp.107-115.
