import matplotlib.pyplot as plt
import torch


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
