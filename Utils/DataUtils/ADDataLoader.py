import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from Utils.DataUtils.LoadingUtils import readImage

IMG_SIZE = 256  # ViT expects 256x256 images

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATED_DIR = PROJECT_ROOT / "generated"
LABELS_FILE = PROJECT_ROOT / "oasis_cross-sectional-5708aa0a98d82080.xlsx"


def load_cdr_labels():
    """Load CDR labels from OASIS spreadsheet."""
    df = pd.read_excel(LABELS_FILE)
    labels = {}
    for _, row in df.iterrows():
        subject_id = row['ID']
        cdr = row['CDR']
        if pd.notna(cdr):
            subject_base = subject_id.replace('_MR1', '')
            labels[subject_base] = float(cdr)
    return labels


def cdr_to_binary(cdr):
    """Convert CDR to binary AD classification: 0 = healthy, 1 = dementia."""
    if cdr == 0:
        return 0
    else:
        return 1


def cdr_to_multiclass(cdr):
    """Convert CDR to multiclass: 0=healthy, 1=very mild, 2=mild+."""
    if cdr == 0:
        return 0
    elif cdr == 0.5:
        return 1
    else:
        return 2


class ADMotionDataset(Dataset):
    """Dataset for AD classification at a specific motion severity level."""

    def __init__(self, subject_indices, motion_level, cdr_labels, binary=True):
        self.motion_level = motion_level
        self.cdr_labels = cdr_labels
        self.binary = binary
        self.samples = []

        motion_dir = GENERATED_DIR / f"M{motion_level}"
        if not motion_dir.exists():
            raise ValueError(f"Motion directory {motion_dir} does not exist. Run GenerateMotion.py first.")

        for subject_id in subject_indices:
            if subject_id not in cdr_labels:
                continue

            cdr = cdr_labels[subject_id]
            label = cdr_to_binary(cdr) if binary else cdr_to_multiclass(cdr)

            for img_file in motion_dir.glob(f"{subject_id}_*.tiff"):
                self.samples.append((str(img_file), label, subject_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, subject_id = self.samples[idx]
        image = readImage(img_path, show=False)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        image = image.astype(np.float32)
        
        # Resize to 256x256 for ViT
        if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        if image.max() > 0:
            image = image / image.max()
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image, label


def get_subject_splits(cdr_labels, test_size=0.3, val_size=0.1, random_state=42):
    """Split subjects (not slices) into train/val/test to avoid data leakage.
    
    With test_size=0.3 and val_size=0.1:
    - 70% train (of which 10% becomes validation) = ~63% train, ~7% val
    - 30% test
    """
    subjects = list(cdr_labels.keys())
    labels = [cdr_to_binary(cdr_labels[s]) for s in subjects]

    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, stratify=labels, random_state=random_state
    )

    train_labels = [cdr_to_binary(cdr_labels[s]) for s in train_subjects]
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_size / (1 - test_size),
        stratify=train_labels, random_state=random_state
    )

    return train_subjects, val_subjects, test_subjects


def create_dataloaders_for_motion_level(motion_level, batch_size=16, binary=True):
    """Create train/val/test dataloaders for a specific motion level."""
    cdr_labels = load_cdr_labels()
    train_subjects, val_subjects, test_subjects = get_subject_splits(cdr_labels)

    train_dataset = ADMotionDataset(train_subjects, motion_level, cdr_labels, binary)
    val_dataset = ADMotionDataset(val_subjects, motion_level, cdr_labels, binary)
    test_dataset = ADMotionDataset(test_subjects, motion_level, cdr_labels, binary)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def create_test_loader_for_motion_level(motion_level, test_subjects, cdr_labels, batch_size=16, binary=True):
    """Create test dataloader for a specific motion level using pre-defined test subjects."""
    test_dataset = ADMotionDataset(test_subjects, motion_level, cdr_labels, binary)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader
