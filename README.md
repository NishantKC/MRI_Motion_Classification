# MRI Motion Artifact Classification

Investigating how synthetic motion artifacts in brain MRI scans affect the accuracy of Alzheimer's Disease (AD) classification using a Vision Transformer (ViT).

## Overview

This project generates synthetic motion artifacts on OASIS brain MRI slices at five severity levels (M0–M4) using Cartesian k-space sampling, then measures the degradation of a ViT-based AD classifier trained on clean (M0) images.

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
