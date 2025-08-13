#!/usr/bin/python3
"""Utils."""
from typing import TypeVar

T = TypeVar('T')


def safe(value: T | None) -> T:
    """Assert value is not None."""
    assert value is not None
    return value
