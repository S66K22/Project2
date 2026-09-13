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
    try:
        with Image.open(path) as img:
            img.verify()

        # Re-open because verify() invalidates the image object
        with Image.open(path) as img:
            img.load()
        ret = True
    except Exception as e:
        ret = False
    return ret


def check_images_format(directory, corrupted_files_dir):

    for path in directory.iterdir():
        if find_file_format(path) != "image/jpeg":
            logger.info(f"File with path {path} is not an image.")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)
        elif not can_image_be_loaded(path):
            logger.info(f"Invalid/corrupted image: {path}")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)