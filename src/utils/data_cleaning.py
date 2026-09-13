import magic
from pathlib import Path


def find_file_format(filepath):
    file_type = magic.from_file(filepath, mime=True)
    return file_type
