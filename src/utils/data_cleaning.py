import logging
from pathlib import Path

import magic

logger = logging.getLogger(__name__)


def move_file(old_path, new_path):
    if not new_path.exists():
        new_path.mkdir(parents=True)

    old_path.rename(new_path / old_path.name)


def find_file_format(filepath):
    file_type = magic.from_file(filepath, mime=True)
    return file_type


def check_images_format(directory, corrupted_files_dir):

    for path in directory.iterdir():
        if find_file_format(path) != "image/jpeg":
            logger.info(f"File with path {path} is not an image.")
            corrupted_files_dir = corrupted_files_dir / path.parent.name
            move_file(path, corrupted_files_dir)

