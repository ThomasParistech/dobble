# /usr/bin/python3
"""Extract the binary mask of the symbols images"""


import copy
import os

import cv2
import numpy as np
from tqdm import tqdm

from utils import new_folder

DEBUG = False


def _get_contour_mask(img: np.ndarray, margin_pix: int, ths: int) -> np.ndarray:
    assert img.shape[0] == img.shape[1]

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < ths
    mask = 255*mask.astype(np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, element, iterations=margin_pix)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    assert len(contours) != 0
    contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    if DEBUG:
        display = copy.deepcopy(img)
        cv2.drawContours(display, [contour], 0, (0, 0, 255), 2)
        cv2.imshow("Contour", display)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)

    return mask


def main(symbols_folder: str,
         out_masks_folder: str,
         computing_size_pix: str = 300,
         low_res_size_pix: str = 100,
         margin_pix: str = 12,
         ths: int = 250):
    """
    Extract the mask of symbols

    Args:
        symbols_folder: Folder containing 57 square colored symbol images with white background
        out_masks_folder: Output folder that will contain the low resolution symbol masks
        computing_size_pix: Size of the images when finding contours and applying dilation
        low_res_size_pix: Output size of the low resolution dumped masks
        margin_pix: Dilation applied around the mask, covariant with computing_size_pix
        ths: Pixels the intensity of which is above this threshold are considered as white background
    """
    names = [f.name for f in os.scandir(symbols_folder)]
    assert len(names) == 57

    new_folder(out_masks_folder)

    for name in tqdm(names, "Find contour masks"):
        input_path = os.path.join(symbols_folder, name)
        output_mask_path = os.path.join(out_masks_folder, name)

        img = cv2.imread(input_path)
        assert img.shape[0] == img.shape[1], "Expect a square image. Consider using 'preprocess.py'"
        assert len(img.shape) == 3, "Expect a colored image"

        resized_img = cv2.resize(img, (computing_size_pix, computing_size_pix))
        resized_mask = _get_contour_mask(resized_img, margin_pix, ths)

        low_res_mask = cv2.resize(resized_mask, (low_res_size_pix,
                                                 low_res_size_pix))
        cv2.imwrite(output_mask_path, low_res_mask)
