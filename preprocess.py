# /usr/bin/python3
"""Make all images square and rotation-proof"""


import math
import os

import cv2
import numpy as np
from tqdm import tqdm

from utils import new_folder


def _to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    s = max(h, w)
    top = int((s-h)/2)
    left = int((s-w)/2)

    # Enlarge the image to contain the circumscribed circle of the original square image
    new_size = math.ceil(np.sqrt(2)*s)
    pad = math.ceil(0.5*(new_size - s))

    square = 255*np.ones((s+2*pad, s+2*pad, 3), dtype=np.uint8)
    top += pad
    left += pad
    square[top:top+h, left:left+w] = img

    return square


def main(images_folder: str,
         out_images_folder: str):
    """
    Make all images square and add white margin to make sure
    the content won't be cropped after a rotation

    Args:
        images_folder: Folder containing colored images to preprocess
        out_images_folder: Output folder containing the square preprocessed images
    """
    names = [f.name for f in os.scandir(images_folder)]

    new_folder(out_images_folder)

    for name in tqdm(names, "Square images"):
        input_path = os.path.join(images_folder, name)
        output_path = os.path.join(out_images_folder, name)

        img = cv2.imread(input_path)
        img = _to_square(img)
        cv2.imwrite(output_path, img)
