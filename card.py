# /usr/bin/python3
"""Generate 57 cards with randomly drawn symbols"""

import math
import os
from dataclasses import dataclass
from typing import List
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from optim import get_cards
from utils import new_folder

DEBUG = False
DEBUG_FINAL = False

RNG = np.random.default_rng(42)

SCALE_TARGETS_LIST = [
    [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    [0.6, 0.8, 0.9, 0.9, 5.0, 5.0, 5.0, 5.0],
    [0.7, 0.7, 0.7, 2, 2., 7., 7., 7.],
    [0.8, 0.8, 0.8, 0.8, 5., 5., 7., 7.]
]
# Account for the added margin around the image
SCALE_TARGETS_LIST = [[np.sqrt(2)*s for s in scales]
                      for scales in SCALE_TARGETS_LIST]

TRANSLATION_MAX_STEP = 30
ANGLE_MAX_STEP = 90

SCALE_RATIO_MIN = 0.6
SCALE_RATIO_MAX = 3.0


SQUARE_SIZE = 1.0/(np.sqrt(2)*3)
XY_INIT_NORMED = [(0.5 + x * SQUARE_SIZE,
                   0.5 + y * SQUARE_SIZE)
                  for x in [-1.5, -0.5, 0.5]
                  for y in [-1.5, -0.5, 0.5]]


def rotate(image: np.ndarray, angle: float, background: Union[Tuple[int, int, int], int]) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=background)

    return rotated


@dataclass
class Symbol:
    ref_mask: np.ndarray
    scale_target: float

    scale: float
    angle: float

    y_top: int
    x_left: int

    def try_params(self, full_mask: np.ndarray, scale: float, x_left: int, y_top: int, angle: float) -> bool:
        """Generate the new mask and return True if it doesn't overlap the neighboring symbols masks"""
        assert len(full_mask.shape) == 2
        full_size = full_mask.shape[0]

        new_size = int(scale*self.ref_mask.shape[0])
        y_bot = y_top+new_size
        x_right = x_left+new_size
        if x_left < 0 or x_right >= full_size or y_top < 0 or y_bot >= full_size:
            return False

        rot = rotate(self.ref_mask, angle, 0)
        resized = cv2.resize(rot, (new_size, new_size))

        n_inter = np.count_nonzero(np.logical_and(resized,
                                                  full_mask[y_top:y_bot, x_left:x_right]))
        return n_inter == 0

    def draw_mask(self, mask: np.ndarray):
        """Superimpose the mask symbol on the input image"""
        assert len(mask.shape) == 2
        rot = rotate(self.ref_mask, self.angle, background=0)
        new_size = int(self.scale*self.ref_mask.shape[0])
        resized = cv2.resize(rot, (new_size, new_size))

        mask[self.y_top:self.y_top+new_size,
             self.x_left:self.x_left+new_size] |= resized


class Card:
    """
    Start with 8 symbols of the same size in the middle of the image card
    and let them evolve overtime to randomly translate, rotate and shrink/grow,
    under the constraint of non-overlapping 

    No need to work with the high-resolution symbol images. Low-res binary masks are
    enough to estimate the overlap.
    """

    def __init__(self, masks: List[np.ndarray]) -> None:
        """Init from a list of 8 low-resolution symbol masks"""
        assert len(masks) == 8

        self.size_pix = math.ceil(masks[0].shape[0] / SQUARE_SIZE)
        self.center = (self.size_pix // 2, self.size_pix // 2)

        self.symbols: List[Symbol] = []
        list_xy = np.array(XY_INIT_NORMED)
        RNG.shuffle(list_xy)
        list_xy = list_xy[:8]

        idx = RNG.integers(0, len(SCALE_TARGETS_LIST))
        scale_targets = SCALE_TARGETS_LIST[idx]
        RNG.shuffle(scale_targets)

        for (x, y), mask, scale_target in zip(list_xy, masks, scale_targets):
            self.symbols.append(
                Symbol(ref_mask=mask,
                       scale=1.0,
                       scale_target=scale_target,
                       angle=RNG.integers(0, 360),
                       y_top=int(y*self.size_pix),
                       x_left=int(x*self.size_pix)))

    def next(self, display: bool = False):
        """Let one symbol randomly evolve"""
        full_mask = 255*np.ones((self.size_pix, self.size_pix), dtype=np.uint8)
        cv2.circle(full_mask, self.center, self.center[0], 0, -1)

        selected_idx = RNG.integers(0, 8)
        for k, symbol in enumerate(self.symbols):
            if k != selected_idx:
                symbol.draw_mask(full_mask)

        s = self.symbols[selected_idx]

        mode = RNG.random()
        if mode < 0.2:  # ANGLE
            for _ in range(100):
                d_angle = RNG.integers(ANGLE_MAX_STEP)
                if s.try_params(full_mask, s.scale, s.x_left, s.y_top, s.angle+d_angle):
                    s.angle += d_angle
                    break
        else:
            ratio_scale = abs(s.scale-s.scale_target)/s.scale_target
            if mode > 0.6 and ratio_scale > 0.05:  # scale
                for new_scale in np.linspace(s.scale_target, s.scale, 30):
                    if s.try_params(full_mask, new_scale, s.x_left, s.y_top, s.angle):
                        s.scale = new_scale
                        break
            else:  # TRANSLATION
                for _ in range(100):
                    dx = RNG.integers(-TRANSLATION_MAX_STEP,
                                      TRANSLATION_MAX_STEP)
                    dy = RNG.integers(-TRANSLATION_MAX_STEP,
                                      TRANSLATION_MAX_STEP)
                    if s.try_params(full_mask, s.scale, s.x_left+dx, s.y_top+dy, s.angle):
                        s.x_left += dx
                        s.y_top += dy
                        break

        if display:
            self.imshow(wait_key=1)

    def imshow(self, wait_key: int = 0):
        """Display the current binary mask"""
        full_mask = 255*np.ones((self.size_pix, self.size_pix), dtype=np.uint8)
        cv2.circle(full_mask, self.center, self.center[0], 0, -1)
        for symbol in self.symbols:
            symbol.draw_mask(full_mask)
        cv2.imshow("Card Mask", full_mask)
        cv2.waitKey(wait_key)


def main(masks_folder: str,
         symbols_folder: str,
         out_cards_folder: str,
         card_size_pix: int = 3000,
         circle_width_pix: int = 3,
         n_iter: int = 1000):
    """
    Generate 57 Dobble cards from symbols masks and images

    Args:
        masks_folder: Folder containing the low-resolution symbols masks images
        symbols_folder: Folder containig the high-resolution symbols colored images
        out_cards_folder: Output folder containing the high-resolution random drawn cards
        card_size_pix: Size of the output high-resolution cards
        circle_width_pix: Width of the circle around each card. Covariant with card_size_pix
        n_iter: Number of evolution steps for each card
    """
    names = [f.name for f in os.scandir(masks_folder)]
    assert len(names) == 57

    cards = get_cards()
    assert len(cards) == 57

    new_folder(out_cards_folder)

    for card_idx, symbols in enumerate(tqdm(cards, "Cards")):
        card_path = os.path.join(out_cards_folder, f"card_{card_idx}.png")

        masks = [cv2.imread(os.path.join(masks_folder, names[symbol_idx]),
                            cv2.IMREAD_GRAYSCALE)
                 for symbol_idx in symbols]

        card = Card(masks)
        for _ in range(n_iter):
            card.next(DEBUG)

        symbols_images = [cv2.imread(os.path.join(symbols_folder, names[symbol_idx]))
                          for symbol_idx in symbols]

        card_img = 255*np.ones((card_size_pix, card_size_pix, 3), np.uint8)
        global_scale = float(card_size_pix) / card.size_pix

        for symbol, symbol_img in zip(card.symbols, symbols_images):
            y_top = int(symbol.y_top*global_scale)
            x_left = int(symbol.x_left*global_scale)
            scale = symbol.scale*global_scale
            new_size = int(scale*symbol.ref_mask.shape[0])

            rot_mask = rotate(symbol.ref_mask, symbol.angle, background=0)
            resized_mask = cv2.resize(rot_mask, (new_size, new_size))

            rot_symbol = rotate(symbol_img, symbol.angle,
                                background=(255, 255, 255))
            resized_symbol = cv2.resize(rot_symbol, (new_size, new_size),
                                        interpolation=cv2.INTER_AREA)

            card_img[y_top:y_top+new_size,
                     x_left:x_left+new_size][resized_mask > 0] = resized_symbol[resized_mask > 0]

        cv2.circle(card_img, (card_size_pix // 2, card_size_pix // 2),
                   card_size_pix // 2, (0, 0, 0), circle_width_pix)
        cv2.imwrite(card_path, card_img)

        if DEBUG_FINAL:
            cv2.imshow("Card Image", card_img)
            cv2.waitKey(1)
