#!/usr/bin/python3
"""File Helpers."""

import glob
import os
import shutil

from dobble.utils.asserts import assert_eq
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


def list_svg_files(images_folder: str) -> list[str]:
    """List SVG files."""
    return [f.name
            for f in os.scandir(images_folder)
            if f.name.lower().endswith('.svg')]


def copy_file(input_path: str, output_path: str) -> None:
    """Copy file or folder in other folder."""
    make_sure_folder_exists(output_path)
    if os.path.isdir(input_path):
        for file_path in glob.glob(os.path.join(input_path, '*')):
            basename = os.path.basename(file_path)
            copy_file(file_path, os.path.join(output_path, basename))
    else:
        _, input_ext = os.path.splitext(input_path)
        _, output_ext = os.path.splitext(output_path)

        assert_eq(input_ext, output_ext, msg="Source and destination files must have the same extension")
        shutil.copy(input_path, output_path)
