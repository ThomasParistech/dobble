# /usr/bin/python3
"""Dobble"""


import os
import shutil
import time
from typing import List
from typing import Sized
from typing import Tuple


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


class LogScopeTime:
    """Log the time spent inside a scope. Use as context `with LogScopeTime(name):`."""

    def __init__(self, name: str):
        self._name = name

    def __enter__(self):
        self._start_time = time.perf_counter()
        print(self._name)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        end_time = time.perf_counter()
        print(f"--- {(end_time - self._start_time): .2f} s ({self._name}) ---")
