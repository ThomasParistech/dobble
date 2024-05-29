# /usr/bin/python3
"""Dobble"""


import os
import shutil
from typing import Sized, List


def new_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)


def assert_len(seq: Sized, size: int):
    """Assert Python list has expected length."""
    assert len(seq) == size, \
        f"Expect sequence of length {size}. Got length {len(seq)}."


def list_image_files(images_folder: str) -> List[str]:
    """List image files."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return [f.name
            for f in os.scandir(images_folder)
            if f.name.lower().endswith(image_extensions)]
