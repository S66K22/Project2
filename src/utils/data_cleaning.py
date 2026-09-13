import magic
from pathlib import Path


def find_file_format(filepath):
    file_type = magic.from_file(filepath, mime=True)
    return file_type

def check_images_format(dir):
    directory = Path(dir)

    for path in directory.iterdir():
        if find_file_format(path) != "image/jpeg":
            print(path)
