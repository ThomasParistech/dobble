#!/usr/bin/python3
"""Numpy helpers for type annotations."""

from typing import Any
from typing import TypeAlias

import numpy as np

AnyImage: TypeAlias = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
NpArrayType: TypeAlias = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any] | np.bool_]]
NpFloatArrayType: TypeAlias = np.ndarray[Any, np.dtype[np.floating[Any]]]
NpIntArrayType: TypeAlias = np.ndarray[Any, np.dtype[np.integer[Any]]]
NpBoolArrayType: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
