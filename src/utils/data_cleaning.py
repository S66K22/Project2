import logging
from pathlib import Path

import magic

logger = logging.getLogger(__name__)


def find_file_format(filepath):
    file_type = magic.from_file(filepath, mime=True)
    return file_type


def check_images_format(dir):
    directory = Path(dir)

    for path in directory.iterdir():
        if find_file_format(path) != "image/jpeg":
            logger.info(f"File with path {path} is not an image.")
