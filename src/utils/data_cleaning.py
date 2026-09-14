import hashlib
import logging
from pathlib import Path

import magic
from PIL import Image

logger = logging.getLogger(__name__)


def file_hash(path, chunk_size=8192):
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


def separate_valid_invalid_images(path, corrupted_files_dir):

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


def image_preprocessing():
    data_dir = Path("data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    unclean_dir = data_dir / "unclean"
    corrupted_files_dir = data_dir / "corrupted"

    for data_dir, crop_dir in zip([train_dir, test_dir, unclean_dir], ['train', 'test', 'unclean']):
        corrupted_file_dir = corrupted_files_dir / crop_dir
        for path in get_file_paths(data_dir): 
            separate_valid_invalid_images(path, corrupted_file_dir)


image_preprocessing()