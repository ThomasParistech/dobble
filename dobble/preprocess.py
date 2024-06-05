# /usr/bin/python3
"""Make all images square and rotation-proof"""


import copy
import glob
import math
import os
from typing import Tuple

import cv2
import imagesize
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dobble.utils import get_overlapping_ranges
from dobble.utils import list_image_files
from dobble.utils import new_folder

DEBUG_MASK = False
DEBUG_ENCLOSING_CIRCLE = False


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


def _get_dilation_element(margin_pix: int) -> Tuple[np.ndarray, int, int]:
    ksize = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    iterations = math.ceil(margin_pix/(ksize-1.))
    return element, iterations, iterations*(ksize - 1)


def _get_contour_mask(img: np.ndarray, computing_size_pix: int, margin_pix: int, ths: int) -> np.ndarray:
    assert img.shape[0] == img.shape[1]

    if img.shape[0] != computing_size_pix:
        img = cv2.resize(img, (computing_size_pix, computing_size_pix))

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < ths
    mask = 255*mask.astype(np.uint8)

    element, iterations, _ = _get_dilation_element(margin_pix)
    mask = cv2.dilate(mask, element, iterations=iterations)

    # Fill interior holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, contours, 255)

    if DEBUG_MASK:
        display = copy.deepcopy(img)
        cv2.drawContours(display, contours, -1, (0, 0, 255), 2)
        cv2.imshow("Contour", display)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)

    return mask


def _to_square(img: np.ndarray, margin_pix: int, mask_computing_size_pix: int) -> np.ndarray:
    h, w = img.shape[:2]

    s = max(h, w)
    top = int((s-h)/2)
    left = int((s-w)/2)

    _, _, actual_margin_pix = _get_dilation_element(margin_pix)
    pad = math.ceil((actual_margin_pix+1) * s / mask_computing_size_pix)

    square = 255*np.ones((s+2*pad, s+2*pad, 3), dtype=np.uint8)
    top += pad
    left += pad
    square[top:top+h, left:left+w] = img

    return square


def _center_around_min_enclosing_circle(img: np.ndarray,
                                        mask: np.ndarray,
                                        mask_low_res_size_pix: int) -> Tuple[np.ndarray, np.ndarray]:
    # Enlarge the image to contain the circumscribed circle of the original square image
    assert img.shape[0] == img.shape[1]
    assert mask.shape[0] == mask.shape[1]

    low_res_mask = cv2.resize(mask, (mask_low_res_size_pix, mask_low_res_size_pix),
                              interpolation=cv2.INTER_NEAREST)

    points = np.argwhere(low_res_mask)[:, ::-1]
    assert len(points) != 0
    (c_x, c_y), radius = cv2.minEnclosingCircle(points)

    scale = img.shape[0] / low_res_mask.shape[0]
    c_x *= scale
    c_y *= scale
    radius *= scale

    new_size = math.ceil(2*radius * 1.1)
    new_image = np.full((new_size, new_size, 3),
                        fill_value=255, dtype=np.uint8)
    offset_y = math.floor(c_y - 0.5*new_size)
    offset_x = math.floor(c_x - 0.5*new_size)

    img_y_begin, img_y_end, new_y_begin, new_y_end = get_overlapping_ranges(img.shape[0],
                                                                            new_size, offset_y)
    img_x_begin, img_x_end, new_x_begin, new_x_end = get_overlapping_ranges(img.shape[0],
                                                                            new_size, offset_x)

    new_image[new_y_begin:new_y_end,
              new_x_begin:new_x_end] = img[img_y_begin:img_y_end,
                                           img_x_begin:img_x_end]

    scale = low_res_mask.shape[0] / img.shape[0]
    new_size = int(new_image.shape[0]*scale)
    new_low_res_mask = np.zeros((new_size, new_size), dtype=np.uint8)
    offset_y = int(offset_y*scale)
    offset_x = int(offset_x*scale)

    img_y_begin, img_y_end, new_y_begin, new_y_end = get_overlapping_ranges(low_res_mask.shape[0],
                                                                            new_size, offset_y)
    img_x_begin, img_x_end, new_x_begin, new_x_end = get_overlapping_ranges(low_res_mask.shape[0],
                                                                            new_size, offset_x)
    new_low_res_mask[new_y_begin:new_y_end,
                     new_x_begin:new_x_end] = low_res_mask[img_y_begin:img_y_end,
                                                           img_x_begin:img_x_end]

    if DEBUG_ENCLOSING_CIRCLE:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        display_mask = cv2.resize(low_res_mask, (img.shape[0], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        display = copy.deepcopy(img)
        cv2.circle(display, (int(c_x), int(c_y)), int(radius), (0, 255, 0), 10)
        plt.imshow(display[..., ::-1])
        plt.imshow(display_mask, alpha=0.3)

        plt.subplot(1, 2, 2)
        new_display_mask = cv2.resize(new_low_res_mask, (new_image.shape[0], new_image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        new_display = copy.deepcopy(new_image)
        cv2.circle(new_display, (int(0.5*new_display.shape[0]), int(0.5*new_display.shape[0])),
                   int(radius), (0, 255, 0), 10)
        plt.imshow(new_display[..., ::-1])
        plt.imshow(new_display_mask, alpha=0.3)

        plt.show()

    return new_image, new_low_res_mask


def main(images_folder: str,
         out_images_folder: str,
         out_masks_folder: str,
         largest_svg_side_pix: int,
         mask_computing_size_pix: str,
         mask_low_res_size_pix: str,
         mask_margin_pix: str,
         mask_ths: int):
    """
    Make all images square and add white margin to make sure
    the content won't be cropped after a rotation

    Args:
        images_folder: Input folder containing colored images to preprocess
        out_images_folder: Output folder containing the square preprocessed images
        largest_svg_side_pix: Size of the largest image side (in pix) when rasterizing a SVG image
        mask_computing_size_pix: Size of the images when finding contours and applying dilation
        mask_low_res_size_pix: Output size of the low resolution dumped masks
        mask_margin_pix: Dilation applied around the mask, covariant with computing_size_pix
        mask_ths: Pixels the intensity of which is above this threshold are considered as white background
    """
    rasterize_svg_images(images_folder, largest_svg_side_pix)

    names = list_image_files(images_folder)

    new_folder(out_masks_folder)
    new_folder(out_images_folder)

    for name in tqdm(names, "Preprocess images"):
        input_path = os.path.join(images_folder, name)
        output_path = os.path.join(out_images_folder, name)
        output_mask_path = os.path.join(out_masks_folder, name)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        img = _set_white_background(img)
        img = _to_square(img, mask_margin_pix, mask_computing_size_pix)

        resized_mask = _get_contour_mask(img, mask_computing_size_pix,
                                         mask_margin_pix, mask_ths)

        img, low_res_mask = _center_around_min_enclosing_circle(img, resized_mask,
                                                                mask_low_res_size_pix)

        cv2.imwrite(output_path, img)
        cv2.imwrite(output_mask_path, low_res_mask)
