import logging
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
# import torchmetrics
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, samples_per_class):
        self.dataset = dataset
        self.samples_per_class = samples_per_class

        # Get labels from the dataset
        labels = [dataset[i][1] for i in range(len(dataset))]

        self.class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = sorted(self.class_to_indices.keys())

    def __iter__(self):
        # Shuffle indices of every class
        class_indices = {
            cls: random.sample(indices, len(indices))
            for cls, indices in self.class_to_indices.items()
        }

        # Number of complete balanced batches
        num_batches = min(
            len(indices) // self.samples_per_class for indices in class_indices.values()
        )

        for batch_idx in range(num_batches):
            batch = []

            for cls in self.classes:
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class

                batch.extend(class_indices[cls][start:end])

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return min(
            len(indices) // self.samples_per_class
            for indices in self.class_to_indices.values()
        )


def create_dataloader(
    path, transformation, samples_per_class=4, val=False, batch_size=32
):
    dataset = ImageFolder(
        path,
        transform=transformation,
    )

    batch_sampler = BalancedBatchSampler(
        dataset,
        samples_per_class=samples_per_class,
    )

    if val:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
        )

    return loader, dataset.classes


def create_cv_loaders(
    path,
    train_transform=None,
    batch_size=32,
    n_splits=5,
    samples_per_class=4,
    random_state=42,
):
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.33),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.05,
                    hue=0.05,
                ),
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

    # Separate datasets because they use different transforms.
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

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_indices, val_indices) in enumerate(
        skf.split(indices, labels),
        start=1,
    ):
        train_dataset = Subset(
            train_full,
            train_indices,
        )

        val_dataset = Subset(
            val_full,
            val_indices,
        )

        batch_sampler = BalancedBatchSampler(
            train_dataset,
            samples_per_class=samples_per_class,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Log distribution for this fold.
        train_counts = Counter(labels[i] for i in train_indices)

        val_counts = Counter(labels[i] for i in val_indices)

        logger.info(f"========== Fold {fold}/{n_splits} ==========")

        for class_idx, class_name in enumerate(train_full.classes):
            logger.info(
                f"{class_name:10s} "
                f"train={train_counts[class_idx]:4d} "
                f"val={val_counts[class_idx]:4d}"
            )

        yield fold, train_loader, val_loader, train_full.classes


def create_train_val_loader(path, batch_size=32, test_size=0.2):
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.05,
                saturation=0.01,
                hue=0.05,
            ),
            # transforms.RandomGrayscale(p=0.05),
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
    batch_sampler = BalancedBatchSampler(
        train_loader.dataset,
        samples_per_class=4,
    )

    balanced_train_loader = DataLoader(
        train_loader.dataset,
        batch_sampler=batch_sampler,
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

    return balanced_train_loader, val_loader, train_full.classes


def create_subset_from_loader(dataloader, num_imgs_per_class=3):
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


def evaluate_tm(
    model,
    data_loader,
    accuracy_metric,
    f1_metric,
    device,
):
    model.eval()

    accuracy_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            accuracy_metric.update(y_pred, y_batch)
            f1_metric.update(y_pred, y_batch)

    accuracy = accuracy_metric.compute()
    macro_f1 = f1_metric.compute()

    return accuracy, macro_f1


def train(
    model,
    optimizer,
    loss_fn,
    accuracy_metric,
    f1_metric,
    train_loader,
    valid_loader,
    n_epochs,
    device,
    loss_type="ce",
    layers_to_eval=None,
    path_to_save="",
    use_scheduler=False,
    cosine_annealing=False,
    patience=10,
    factor=0.1,
):
    if use_scheduler:
        if not cosine_annealing:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=patience,
                factor=factor,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=1e-6,
            )

    history = {
        "train_losses": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "valid_accuracy": [],
        "valid_macro_f1": [],
        "learning_rates": [],
    }

    logger.debug("=" * 70)

    best_val_macro_f1 = 0.0
    for epoch in range(n_epochs):
        total_loss = 0.0
        accuracy_metric.reset()
        f1_metric.reset()
        model.train()
        if layers_to_eval is not None:
            layers_to_eval.eval()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            if loss_type == "ce":
                loss = loss_fn(y_pred, y_batch)

            elif loss_type == "bce":
                targets_one_hot = F.one_hot(
                    y_batch,
                    num_classes=8
                ).float()

                loss = loss_fn(y_pred, targets_one_hot)

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}"
                )

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy_metric.update(y_pred, y_batch)
            f1_metric.update(y_pred, y_batch)

        train_accuracy = accuracy_metric.compute().item()
        train_macro_f1 = f1_metric.compute().item()

        history["train_losses"].append(
            total_loss / len(train_loader)
        )

        history["train_accuracy"].append(train_accuracy)
        history["train_macro_f1"].append(train_macro_f1)

        val_accuracy, val_macro_f1 = evaluate_tm(
            model,
            valid_loader,
            accuracy_metric,
            f1_metric,
            device,
        )

        val_accuracy = val_accuracy.item()
        val_macro_f1 = val_macro_f1.item()

        history["valid_accuracy"].append(val_accuracy)
        history["valid_macro_f1"].append(val_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        
        history["learning_rates"].append(current_lr)
        # lr_string = ", ".join(f"{current_lr:.2e}")
        
        if use_scheduler:
            if not cosine_annealing:
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        logger.debug(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
            f"train acc: {train_accuracy:.4f}, "
            f"train macro F1: {train_macro_f1:.4f}, "
            f"val acc: {val_accuracy:.4f}, "
            f"val macro F1: {val_macro_f1:.4f}, "
            f"lr: {current_lr}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            if len(path_to_save) > 0:
                torch.save(
                    model.state_dict(),
                    path_to_save
                )
    return history


def extract_features(loader, model, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, f = model(x.to(device), return_features=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def calc_centroids_cov_inv(train_feats, train_labels, num_classes=8):
    centroids = np.stack(
        [train_feats[train_labels == c].mean(axis=0) for c in range(num_classes)]
    )

    centered = np.concatenate(
        [train_feats[train_labels == c] - centroids[c] for c in range(num_classes)]
    )
    cov = np.cov(centered, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6  # regularize for invertibility
    cov_inv = np.linalg.inv(cov)

    return centroids, cov_inv


def mahalanobis_min_distance(feat_vec, centroids, cov_inv):
    diffs = feat_vec[None, :] - centroids
    dists = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)  # squared distance
    return dists.min(), dists.argmin()


def fit_temperature(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_list.append(model(x.to(device)).cpu())
            labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()
