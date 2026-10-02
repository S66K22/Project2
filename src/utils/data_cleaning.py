import hashlib
import logging
from pathlib import Path

import magic
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Load once, not every time the function is called
yolo_model = YOLO("yolo11n.pt")


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_file_hash(path, chunk_size=8192):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def move_file(old_path, new_path):
    if not new_path.exists():
        new_path.mkdir(parents=True)

    old_path.rename(new_path / old_path.name)


def find_file_format(filepath):
    file_type = magic.from_file(filepath, mime=True)
    return file_type


def can_image_be_loaded(path):
    ret = False
    width, height = 0, 0
    try:
        with Image.open(path) as img:
            img.verify()

        # Re-open because verify() invalidates the image object
        with Image.open(path) as img:
            img.load()
            width, height = img.size
        ret = True
    except Exception as e:
        ret = False
        width, height = 0, 0
    return ret, width, height


def is_image_size_valid(height, width):
    ret = True
    if height < 32 or width < 32:
        ret = False
    ratio = width / height
    if ratio < 0.33 or ratio > 3.0:
        ret = False
    return ret


def separate_valid_invalid_images(path, corrupted_files_dir, hashes):
    file_hash = get_file_hash(path)
    if file_hash in hashes:
        logger.info(
            f"File with path {path} has same hash with file {hashes[file_hash]}."
        )
        corrupted_files_dir = corrupted_files_dir / path.parent.name
        move_file(path, corrupted_files_dir)
        return
    else:
        hashes[file_hash] = path

    if find_file_format(path) != "image/jpeg":
        logger.info(f"File with path {path} is not an image.")
        corrupted_files_dir = corrupted_files_dir / path.parent.name
        move_file(path, corrupted_files_dir)

    else:
        is_valid, width, height = can_image_be_loaded(path)
        if not is_valid or not is_image_size_valid(height, width):
            logger.info(f"Invalid/corrupted image: {path}")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)


def get_file_paths(directory):
    return [path for path in directory.rglob("*") if path.is_file()]


def extract_vehicles(
    image_path,
    model=yolo_model,
    conf_threshold=0.5,
    padding=0.0,
):
    """
    Detect vehicles in an image and return cropped vehicle images.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.

    model : YOLO
        Loaded YOLO model.

    conf_threshold : float
        Minimum detection confidence.

    padding : float
        Fractional padding around each bounding box.
        Example: 0.1 adds 10% padding.

    Returns
    -------
    list[dict]
        Each element contains:
            - image: PIL.Image
            - class_id: YOLO class ID
            - class_name: vehicle class name
            - confidence: detection confidence
            - bbox: (x1, y1, x2, y2)
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False,
    )

    vehicles = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            # Ignore non-vehicle classes
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Add optional padding
            box_width = x2 - x1
            box_height = y2 - y1

            pad_x = box_width * padding
            pad_y = box_height * padding

            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x))
            y2 = min(height, int(y2 + pad_y))

            crop = image.crop((x1, y1, x2, y2))

            vehicles.append(
                {
                    "image": crop,
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )

    return vehicles


def image_preprocessing():
    data_dir = Path("data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    unclean_dir = data_dir / "unclean"
    cropped_dir = data_dir / "cropped"
    corrupted_files_dir = data_dir / "corrupted"
    hashes = dict()

    for data_dir, crop_dir in zip(
        [train_dir, test_dir, unclean_dir, cropped_dir],
        ["train", "test", "unclean", "cropped"],
    ):
        corrupted_file_dir = corrupted_files_dir / crop_dir
        for path in get_file_paths(data_dir):
            separate_valid_invalid_images(path, corrupted_file_dir, hashes)


def extrac_vehicle_from_dir():
    data_dir = Path("data")
    train_dir = data_dir / "train"
    cropped_dir = data_dir / "cropped"

    if not cropped_dir.exists():
        cropped_dir.mkdir(parents=True, exist_ok=True)

    for path in get_file_paths(train_dir):
        vehicles = extract_vehicles(path, conf_threshold=0.1)
        for i, vehicle in enumerate(vehicles):
            logger.debug(
                i,
                vehicle["class_name"],
                vehicle["confidence"],
                vehicle["bbox"],
            )
            image_name = path.name
            image_dir = cropped_dir / path.parent.name
            if not image_dir.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
            vehicle["image"].save(image_dir / image_name)
