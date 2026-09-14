import logging
from pathlib import Path

import magic
from PIL import Image

logger = logging.getLogger(__name__)


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
    if ratio < 0.5 or ratio > 2.0:
        ret = False
    return ret


def separate_valid_invalid_images(directory, corrupted_files_dir):

    for path in directory.iterdir():
        if find_file_format(path) != "image/jpeg":
            logger.info(f"File with path {path} is not an image.")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)
            continue

        is_valid, width, height = can_image_be_loaded(path)
        if not is_valid:
            logger.info(f"Invalid/corrupted image: {path}")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)
            continue
        if is_image_size_valid(height, width):
            logger.info(f"Invalid/corrupted image: {path}")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)

