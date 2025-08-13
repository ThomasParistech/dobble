#!/usr/bin/python3
"""File Helpers."""

import os
import shutil

from dobble.utils.logger import logger


def remove_folder(folder: str, warning: bool = True) -> None:
    """Remove folder."""
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    elif warning:
        logger.warning(f"Folder {folder} doesn't exist")


def make_sure_folder_exists(path: str) -> None:
    """Ensure the folder for a path exists. Use parent if path looks like a file."""
    folder = os.path.dirname(path) if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)


def create_new_folder(folder: str) -> None:
    """Create a new folder and clean it if it already exists."""
    remove_folder(folder, warning=False)
    os.makedirs(folder)


def list_image_files(images_folder: str) -> list[str]:
    """List image files."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return [f.name
            for f in os.scandir(images_folder)
            if f.name.lower().endswith(image_extensions)]
