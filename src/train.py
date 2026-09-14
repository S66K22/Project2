import logging
from collections import Counter

import torch
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def create_train_val_loader(path, batch_size=32, test_size=0.2):
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    train_full = ImageFolder(
        "data/train",
        transform=train_transform,
    )

    val_full = ImageFolder(
        "data/train",
        transform=val_transform,
    )

    labels = train_full.targets
    indices = list(range(len(train_full)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    train_counts = Counter(train_full.targets[i] for i in train_indices)
    val_counts = Counter(train_full.targets[i] for i in val_indices)

    for class_idx, class_name in enumerate(train_full.classes):
        logger.info(
            f"{class_name:10s} "
            f"train={train_counts[class_idx]:4d} "
            f"val={val_counts[class_idx]:4d}"
        )

    return train_loader, val_loader
