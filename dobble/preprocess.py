# /usr/bin/python3
"""Make all images square and rotation-proof"""


from dobble.utils import assert_len, list_image_files
import matplotlib.pyplot as plt
import math
import os

import cv2
import numpy as np
from tqdm import tqdm
from typing import List

from dobble.utils import new_folder

import glob
import cairosvg

import imagesize


def rasterize_svg_images(images_folder: str, largest_side_pix: int) -> None:
    """Rasterize SVG images."""
    svg_files = glob.glob(os.path.join(images_folder, '*.svg'))

    def convert_svg_to_png(in_path: str, out_path: str, set_width: bool):
        option = "output-width" if set_width else "output-height"
        cmd = f"cairosvg '{in_path}' -o '{out_path}' --{option} {largest_side_pix}"
        os.system(cmd)

    for in_path in tqdm(svg_files, desc="SVG to PNG"):
        out_path = in_path.replace('.svg', '.png')
        convert_svg_to_png(in_path, out_path, set_width=True)
        width, height = imagesize.get(out_path)
        if height > width:
            convert_svg_to_png(in_path, out_path, set_width=False)


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


def _set_white_background(img: np.ndarray) -> np.ndarray:
    """Handle alpha channel and set white background."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[-1] == 3:
        return img

    assert img.shape[-1] == 4

    alpha = img[..., -1][..., None] / 255.0
    img = img[..., :3]

    img = (1.0 - alpha) * 255.0 + alpha*img
    img = img.astype(np.uint8)

    return img


def main(images_folder: str,
         out_images_folder: str,
         largest_svg_side_pix: int):
    """
    Make all images square and add white margin to make sure
    the content won't be cropped after a rotation

    Args:
        images_folder: Input folder containing colored images to preprocess
        out_images_folder: Output folder containing the square preprocessed images
        largest_svg_side_pix: Size of the largest image side (in pix) when rasterizing a SVG image
    """
    rasterize_svg_images(images_folder, largest_svg_side_pix)

    names = list_image_files(images_folder)

    new_folder(out_images_folder)

    for name in tqdm(names, "Square images"):
        input_path = os.path.join(images_folder, name)
        output_path = os.path.join(out_images_folder, name)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        img = _set_white_background(img)
        img = _to_square(img)
        cv2.imwrite(output_path, img)
