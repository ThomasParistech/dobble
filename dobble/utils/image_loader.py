#!/usr/bin/python3
"""Load color/gray images."""
import os
from enum import auto
from enum import IntEnum
from typing import cast

import cv2
import numpy as np
from typing_extensions import assert_never

from dobble.utils.asserts import assert_eq
from dobble.utils.asserts import assert_in
from dobble.utils.asserts import assert_isfile
from dobble.utils.asserts import assert_np_shape
from dobble.utils.file import make_sure_folder_exists
from dobble.utils.np_types import NpArrayType
from dobble.utils.np_types import NpIntArrayType
from dobble.utils.utils import safe


def write_image(path: str, img: NpArrayType) -> None:
    """Write image."""
    make_sure_folder_exists(path)
    _, ext = os.path.splitext(path)
    assert_in(ext, ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    assert_eq(img.dtype, np.uint8)

    cv2.imwrite(path, cast(NpIntArrayType, img))


class ImreadType(IntEnum):
    """Image loading type."""
    BGR = auto()
    GRAY = auto()
    BGR_WHITE_BACKGROUND = auto()


def load_image(path: str, mode: ImreadType = ImreadType.BGR) -> NpIntArrayType:
    """Load image."""
    assert_isfile(path)
    if mode is ImreadType.BGR:
        img = safe(cv2.imread(path, cv2.IMREAD_COLOR))
        assert_np_shape(img, (None, None, 3))
    elif mode is ImreadType.GRAY:
        img = safe(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        assert_np_shape(img, (None, None))
    elif mode is ImreadType.BGR_WHITE_BACKGROUND:
        img = safe(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        assert_eq(img.dtype, np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 3:
            assert_np_shape(img, (None, None, 3))
        else:
            assert_np_shape(img, (None, None, 4))
            alpha = img[..., -1][..., None] / 255.0
            img = img[..., :3]
            img = ((1.0 - alpha) * 255.0 + alpha*img).astype(np.uint8)
        assert_np_shape(img, (None, None, 3))
    else:
        assert_never(mode)

    return cast(NpIntArrayType, img)
