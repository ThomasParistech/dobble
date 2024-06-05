# /usr/bin/python3
"""Dobble"""


import os
import shutil
from typing import List
from typing import Sized
from typing import Tuple

import numpy as np


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


def get_overlapping_ranges(len_a: int, len_b: int, offset: int) -> Tuple[int, int, int, int]:
    """Get valid index range when overlapping two lists A and B.

    offset is the coordinate of the begin of B inside A (Could be negative)

    We can then compare a[a_begin:a_end] and b[b_begin:b_end]
    """
    a_begin = max(offset, 0)
    a_end = min(offset + len_b, len_a)
    b_begin = a_begin - offset
    b_end = a_end - offset
    assert a_end - a_begin == b_end - b_begin
    return a_begin, a_end, b_begin, b_end


def get_overlapping_image_ranges(img_a: np.ndarray,
                                 img_b: np.ndarray,
                                 *,
                                 x_left: int,
                                 y_top: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get valid index range when overlapping two images A and B.

    (y_top, x_left) is the coordinates of the top-left corner of B inside A (Could be negative)

    We can then compare cropped_a and cropped_b
    """
    a_y_begin, a_y_end, b_y_begin, b_y_end = get_overlapping_ranges(img_a.shape[0], img_b.shape[0],
                                                                    y_top)
    a_x_begin, a_x_end, b_x_begin, b_x_end = get_overlapping_ranges(img_a.shape[1], img_b.shape[1],
                                                                    x_left)
    cropped_a = img_a[a_y_begin:a_y_end, a_x_begin:a_x_end]
    cropped_b = img_b[b_y_begin:b_y_end, b_x_begin:b_x_end]

    return cropped_a, cropped_b
