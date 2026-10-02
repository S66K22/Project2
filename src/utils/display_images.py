import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def denormalize(images, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    means = torch.tensor(means, device=images.device).view(1, 3, 1, 1)
    stds = torch.tensor(stds, device=images.device).view(1, 3, 1, 1)
    return (images * stds + means).clamp(0, 1)


def ims_show(images, labels, class_names, n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    denormalized_images = denormalize(images)

    for ax, image, label in zip(
        axes.flat, denormalized_images[: n_rows * n_cols], labels[: n_rows * n_cols]
    ):
        # CHW -> HWC
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(class_names[label.item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    model,
    weights_path,
    dataloader,
    class_names,
    device=None,
):
    """
    Plot raw-count and row-normalized confusion matrices side by side.

    Left:
        Raw number of predictions.

    Right:
        Row-normalized percentages. Each row represents one true class
        and sums to 100%.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    # Inference
    with torch.inference_mode():

        for images, labels in dataloader:

            images = images.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Raw confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names)),
    )

    # Row-normalized confusion matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Convert to percentage
    cm_percentage = cm_normalized * 100

    # Plot
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    # -------------------------
    # Raw confusion matrix
    # -------------------------
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )

    axes[0].set_title("Confusion Matrix — Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # -------------------------
    # Normalized confusion matrix
    # -------------------------
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )

    axes[1].set_title("Confusion Matrix — Normalized (%)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.show()

    return cm, cm_percentage


def plot_misclassified_images(
    model,
    weights_path,
    dataloader,
    class_names,
    num_images=16,
    device=None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Plot misclassified images with their true and predicted labels.

    Parameters
    ----------
    model : torch.nn.Module
        Model architecture.

    weights_path : str
        Path to the saved model weights.

    dataloader : DataLoader
        DataLoader containing the evaluation/test dataset.

    class_names : list[str]
        Class names indexed according to the dataset labels.

    num_images : int
        Maximum number of misclassified images to display.

    device : torch.device, optional
        Device used for inference.

    mean : tuple
        Mean used for image normalization.

    std : tuple
        Standard deviation used for image normalization.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load weights
    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []

    with torch.inference_mode():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            # Find incorrect predictions
            incorrect = predictions != labels

            for image, true_label, pred_label in zip(
                images[incorrect],
                labels[incorrect],
                predictions[incorrect],
            ):
                misclassified_images.append(image.cpu())

                misclassified_true.append(true_label.item())

                misclassified_pred.append(pred_label.item())

                # Stop once we have enough
                if len(misclassified_images) >= num_images:
                    break

            if len(misclassified_images) >= num_images:
                break

    # No mistakes
    if len(misclassified_images) == 0:
        print("No misclassified images found.")
        return

    # Number of images actually available
    n = len(misclassified_images)

    # Grid size
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows),
    )

    # Make axes iterable even when there is only one row
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    for ax, image, true_label, pred_label in zip(
        axes,
        misclassified_images,
        misclassified_true,
        misclassified_pred,
    ):

        # Undo normalization
        image = image * std + mean

        # Convert CHW -> HWC
        image = image.permute(1, 2, 0)

        # Keep values in valid display range
        image = image.clamp(0, 1)

        ax.imshow(image)

        ax.set_title(
            f"True: {class_names[true_label]}\n" f"Pred: {class_names[pred_label]}"
        )

        ax.axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
