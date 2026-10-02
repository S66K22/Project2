import torchvision.transforms.v2 as transforms
import torch
from .train import mahalanobis_min_distance
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UNKNOWN_LABEL = "unknown"
preprocess = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False,)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

def predict(image_path, model, ckpt, device):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, feat = model(x, return_features=True)
        probs = torch.softmax(logits / ckpt["temperature"], dim=1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    max_prob = float(probs.max())

    dist, _ = mahalanobis_min_distance(
        feat[0].cpu().numpy(), ckpt["centroids"], ckpt["cov_inv"]
    )

    is_unknown = (dist > ckpt["dist_threshold"]) or (max_prob < ckpt["conf_threshold"])
    label = UNKNOWN_LABEL if is_unknown else ckpt["class_names"][pred_class]

    return {
        "label": label,
        "softmax_conf": max_prob,
        "mahalanobis_dist": float(dist),
        "raw_pred_class": ckpt["class_names"][pred_class],
    }