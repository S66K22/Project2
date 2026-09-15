import logging
from collections import Counter

import numpy as np
import torch
import torchmetrics
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
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_full = ImageFolder(
        path,
        transform=train_transform,
    )

    val_full = ImageFolder(
        path,
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

    return train_loader, val_loader, train_full.classes


def create_subset_from_loader(dataloader):
    dataset = dataloader.dataset

    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    indices = []

    for class_id in range(8):
        class_indices = np.where(labels == class_id)[0]
        indices.extend(class_indices[:3])  # 3 images per class

    small_dataset = Subset(dataset, indices)

    small_loader = DataLoader(
        small_dataset,
        batch_size=len(small_dataset),
        shuffle=True,
    )
    return small_loader


def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train(
    model,
    optimizer,
    loss_fn,
    metric,
    train_loader,
    valid_loader,
    n_epochs,
    device,
    patience=10,
    factor=0.1,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        val_metric = evaluate_tm(model, valid_loader, metric, device).item()
        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
            f"train metric: {history['train_metrics'][-1]:.4f}, "
            f"valid metric: {history['valid_metrics'][-1]:.4f}"
        )
    return history
