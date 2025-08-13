# /usr/bin/python3
"""Make all rasterized images square and rotation-proof."""
import copy
import math
import os
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dobble.utils.asserts import assert_eq
from dobble.utils.asserts import assert_np_shape
from dobble.utils.file import create_new_folder
from dobble.utils.file import list_image_files
from dobble.utils.image_loader import ImreadType
from dobble.utils.image_loader import load_image
from dobble.utils.image_loader import write_image
from dobble.utils.np_types import NpIntArrayType
from dobble.utils.overlapping_ranges import get_overlapping_image_ranges
from dobble.utils.overlapping_ranges import get_overlapping_ranges
from dobble.utils.profiling import profile

DEBUG_MASK = False
DEBUG_ENCLOSING_CIRCLE = False


def _get_dilation_element(margin_pix: int) -> tuple[NpIntArrayType, int, int]:
    ksize = 3
    element = cast(NpIntArrayType, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    iterations = math.ceil(margin_pix/(ksize-1.))
    return element, iterations, iterations*(ksize - 1)


def _get_contour_mask(img: np.ndarray, computing_size_pix: int, margin_pix: int, ths: int) -> NpIntArrayType:
    assert img.shape[0] == img.shape[1]

    if img.shape[0] != computing_size_pix:
        img = cv2.resize(img, (computing_size_pix, computing_size_pix))

    mask = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < ths).astype(np.uint8)

    element, iterations, _ = _get_dilation_element(margin_pix)
    mask = np.asarray(cv2.dilate(mask, element, iterations=iterations))

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


def _to_square(img: NpIntArrayType, margin_pix: int, mask_computing_size_pix: int) -> NpIntArrayType:
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
                                        mask_low_res_size_pix: int) -> tuple[np.ndarray, np.ndarray]:
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

    cropped_img, cropped_new_image = get_overlapping_image_ranges(img, new_image,
                                                                  x_left=offset_x, y_top=offset_y)

    cropped_new_image[...] = cropped_img

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

    new_low_res_mask = cv2.resize(new_low_res_mask, (mask_low_res_size_pix, mask_low_res_size_pix),  # type: ignore
                                  interpolation=cv2.INTER_NEAREST)

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


@profile
def main(images_folder: str,
         out_images_folder: str,
         out_masks_folder: str,
         mask_computing_size_pix: int,
         mask_low_res_size_pix: int,
         mask_margin_pix: int,
         mask_ths: int) -> None:
    """Make all rasterized images square and add white margin to make sure content won't be cropped after a rotation.

    Args:
        images_folder: Input folder containing rasterized colored images to preprocess
        out_images_folder: Output folder containing the square preprocessed images
        out_masks_folder: Output folder containing the masks of the preprocessed images
        largest_svg_side_pix: Size of the largest image side (in pix) when rasterizing a SVG image
        mask_computing_size_pix: Size of the images when finding contours and applying dilation
        mask_low_res_size_pix: Output size of the low resolution dumped masks
        mask_margin_pix: Dilation applied around the mask, covariant with computing_size_pix
        mask_ths: Pixels the intensity of which is above this threshold are considered as white background
    """
    names = list_image_files(images_folder)

    create_new_folder(out_masks_folder)
    create_new_folder(out_images_folder)

    for name in tqdm(names, "Preprocess images"):
        input_path = os.path.join(images_folder, name)
        output_path = os.path.join(out_images_folder, name)
        output_mask_path = os.path.join(out_masks_folder, name)

        img = load_image(input_path, ImreadType.BGR_WHITE_BACKGROUND)
        img = _to_square(img, mask_margin_pix, mask_computing_size_pix)

        resized_mask = _get_contour_mask(img, mask_computing_size_pix,
                                         mask_margin_pix, mask_ths)

        img, low_res_mask = _center_around_min_enclosing_circle(img, resized_mask,
                                                                mask_low_res_size_pix)

        assert_eq(img.shape[0], img.shape[1])
        assert_np_shape(low_res_mask, (mask_low_res_size_pix, mask_low_res_size_pix))
        write_image(output_path, img)
        write_image(output_mask_path, low_res_mask)
