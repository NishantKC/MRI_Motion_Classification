import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from Utils.DataUtils.DataGenerator import getImageAndClasses, wholeIndex


class MRIMotionDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = getImageAndClasses(self.indices[idx])
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)  # Add channel dimension (1, H, W)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.float32))
        return image, label


def create_dataloaders(batch_size=16, test_size=0.3, val_split=0.33):
    labels = np.zeros((1, wholeIndex[-1]), dtype=int)[0]
    for i in range(len(wholeIndex) - 1):
        labels[range(wholeIndex[i], wholeIndex[i + 1])] = int(i)

    all_indices = list(range(wholeIndex[-1]))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        all_indices, labels, test_size=test_size, stratify=labels, shuffle=True
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=val_split, stratify=temp_labels, shuffle=True
    )

    train_dataset = MRIMotionDataset(train_indices)
    val_dataset = MRIMotionDataset(val_indices)
    test_dataset = MRIMotionDataset(test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
