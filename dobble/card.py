# /usr/bin/python3
"""Generate 57 cards with randomly drawn symbols"""

import copy
import math
import numbers
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from dobble.optim import get_cards
from dobble.profiling import profile
from dobble.utils import assert_len
from dobble.utils import get_overlapping_image_ranges
from dobble.utils import list_image_files
from dobble.utils import multiprocess
from dobble.utils import new_folder

DEBUG = False
DEBUG_FINAL = False

RNG = np.random.default_rng(42)

SCALE_TARGETS_LIST: List[List[float]] = [
    [0.8, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    [0.8, 0.8, 0.9, 0.9, 5.0, 5.0, 5.0, 5.0],
    [0.8, 0.8, 0.8, 2, 2., 5., 5., 5.]
]
# Account for the added margin around the image
SCALE_TARGETS_LIST = [[np.sqrt(2)*s for s in scales]
                      for scales in SCALE_TARGETS_LIST]

DISK_OCCUPANCY_TARGET = 0.85


SQUARE_SIZE = 1.0/(np.sqrt(2)*3)
XY_INIT_NORMED = [(0.5 + x * SQUARE_SIZE,
                   0.5 + y * SQUARE_SIZE)
                  for x in [-1.5, -0.5, 0.5]
                  for y in [-1.5, -0.5, 0.5]]


def init_translation(radii: List[int], list_n_angles: List[int]) -> List[Tuple[int, int]]:
    return [(int(np.cos(angle)*r), int(np.sin(angle)*r))
            for r, n_angles in zip(radii, list_n_angles)
            for angle in np.linspace(0, 2*np.pi, endpoint=False, num=n_angles)]


N_ANGLES_TO_TEST = 10
TRANSLATIONS = init_translation(radii=[40, 20, 10], list_n_angles=[7, 5, 4])


@profile
def rotate(image: np.ndarray, angle: int, background: Union[Tuple[int, int, int], int]) -> np.ndarray:
    assert isinstance(angle, numbers.Integral), angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=background)

    return rotated


class CachedTransformedMask:
    """Class used to cache rotated and scaled versions of the mask."""

    def __init__(self, ref_mask: np.ndarray) -> None:
        self.ref_mask = ref_mask

        self.last_angle: int = 0
        self.last_scale: float = 1.0
        self.last_mask = copy.deepcopy(ref_mask)

        @lru_cache(maxsize=None)
        def get_rotated_image(angle: int) -> np.ndarray:
            """Cache up to 360 rotated images."""
            return rotate(self.ref_mask, angle, 0)

        self.get_rotated_image = get_rotated_image

    def get_image(self, angle: int, scale: float) -> np.ndarray:
        """Compute image if not already in cache."""
        if angle != self.last_angle or scale != self.last_scale:
            rot = self.get_rotated_image(angle)
            new_size = int(scale*self.ref_mask.shape[0])
            resized = cv2.resize(rot, (new_size, new_size))

            self.last_angle = angle
            self.last_scale = scale
            self.last_mask = resized

        return self.last_mask


@dataclass
class Symbol:
    ref_mask: np.ndarray
    scale_target: float

    angle: int
    scale: float

    y_top: int
    x_left: int

    def __post_init__(self):
        self.cached_mask = CachedTransformedMask(self.ref_mask)

    @profile
    def try_scale(self, full_mask: np.ndarray) -> Optional[float]:
        """Apply bisection to find largest possible scale."""
        if self.scale_target < self.scale:
            return self.scale_target  # No need to check to reduce the size

        scale_min = self.scale
        scale_max = self.scale_target

        if scale_min == scale_max:
            return None

        if self.try_params(full_mask, scale=scale_max, x_left=self.x_left, y_top=self.y_top, angle=self.angle):
            return scale_max

        largest_possible_scale = -np.inf
        n_steps = max(2, math.ceil(np.log2((scale_max-scale_min)/0.1)))
        for _ in range(n_steps):
            scale_mid = 0.5*(scale_min+scale_max)
            if self.try_params(full_mask, scale=scale_mid, x_left=self.x_left, y_top=self.y_top, angle=self.angle):
                largest_possible_scale = max(scale_mid, largest_possible_scale)
                scale_min = scale_mid
            else:
                scale_max = scale_mid

        return largest_possible_scale if np.isfinite(largest_possible_scale) else None

    def try_params(self, full_mask: np.ndarray, *, scale: float, x_left: int, y_top: int, angle: int) -> bool:
        """Generate the new mask and return True if it doesn't overlap the neighboring symbols masks"""
        assert len(full_mask.shape) == 2
        full_size = full_mask.shape[0]

        new_size = int(scale*self.ref_mask.shape[0])
        y_bot = y_top+new_size
        x_right = x_left+new_size

        # Check if symbol image is completely outside the full mask
        if x_right <= 0 or x_left >= full_size-1 or y_bot <= 0 or y_top >= full_size-1:
            return False

        resized = self.cached_mask.get_image(angle, scale)
        assert resized.shape == (new_size, new_size)

        cropped_mask, cropped_img = get_overlapping_image_ranges(full_mask, resized,
                                                                 x_left=x_left, y_top=y_top)

        # Check that all the True pixels are kept
        if np.count_nonzero(cropped_img) != np.count_nonzero(resized):
            return False

        n_inter = np.count_nonzero(np.logical_and(cropped_img, cropped_mask))
        return n_inter == 0

    @profile
    def draw_mask(self, mask: np.ndarray):
        """Superimpose the mask symbol on the input image"""
        assert len(mask.shape) == 2

        resized = self.cached_mask.get_image(self.angle, self.scale)

        cropped_mask, cropped_img = get_overlapping_image_ranges(mask, resized,
                                                                 x_left=self.x_left, y_top=self.y_top)

        cropped_mask |= cropped_img


class Card:
    """
    Start with 8 symbols of the same size in the middle of the image card
    and let them evolve overtime to randomly translate, rotate and shrink/grow,
    under the constraint of non-overlapping

    No need to work with the high-resolution symbol images. Low-res binary masks are
    enough to estimate the overlap.
    """

    def __init__(self, masks: List[np.ndarray],
                 scale_targets: List[float]) -> None:
        """Init from a list of 8 low-resolution symbol masks"""
        assert len(masks) == 8

        self.size_pix = math.ceil(masks[0].shape[0] / SQUARE_SIZE)
        self.center = (self.size_pix // 2, self.size_pix // 2)

        self.symbols: List[Symbol] = []
        list_xy = np.array(XY_INIT_NORMED)
        RNG.shuffle(list_xy)
        list_xy = list_xy[:8]

        total_target_area = sum(scale_target*np.count_nonzero(mask)
                                for mask, scale_target in zip(masks, scale_targets))
        disk_area = np.pi * self.size_pix**2 / 4
        occupancy = total_target_area / disk_area

        for (x, y), mask, scale_target in zip(list_xy, masks, scale_targets):
            self.symbols.append(
                Symbol(ref_mask=mask,
                       scale=1.0,
                       scale_target=scale_target * DISK_OCCUPANCY_TARGET / occupancy,
                       angle=RNG.integers(0, 360),
                       y_top=int(y*self.size_pix),
                       x_left=int(x*self.size_pix)))

    @profile
    def next(self):
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
            angles = RNG.choice(np.arange(360, step=2),
                                size=N_ANGLES_TO_TEST, replace=False)
            for d_angle in angles:
                if s.try_params(full_mask, scale=s.scale, x_left=s.x_left,
                                y_top=s.y_top, angle=s.angle+d_angle):
                    s.angle += d_angle
                    break
        else:
            ratio_scale = abs(s.scale-s.scale_target)/s.scale_target
            if mode > 0.6 and ratio_scale > 0.05:  # SCALE
                scale = s.try_scale(full_mask)
                if scale is not None:
                    s.scale = scale
            else:  # TRANSLATION
                for dx, dy in TRANSLATIONS:
                    if s.try_params(full_mask, scale=s.scale, x_left=s.x_left+dx,
                                    y_top=s.y_top+dy, angle=s.angle):
                        s.x_left += dx
                        s.y_top += dy
                        break

        if DEBUG:
            self.imshow(wait_key=1)

    def imshow(self, wait_key: int = 0):
        """Display the current binary mask"""
        full_mask = 255*np.ones((self.size_pix, self.size_pix), dtype=np.uint8)
        cv2.circle(full_mask, self.center, self.center[0], 0, -1)
        for symbol in self.symbols:
            symbol.draw_mask(full_mask)
        cv2.imshow("Card Mask", full_mask)
        cv2.waitKey(wait_key)


def allocate_scale_targets(cards: List[List[int]]) -> List[List[float]]:
    """Allocate scale targets while ensuring that each symbol appears at least once with a large scale."""
    scales_per_symbol = [[] for _ in range(57)]

    def compute_score(scales: List[float]) -> float:
        if len(scales) == 0:
            return -np.inf
        return np.max(scales) * np.mean(scales)

    all_scale_targets: List[List[float]] = []
    for symbols in cards:

        idx = RNG.integers(0, len(SCALE_TARGETS_LIST))
        scale_targets = np.array(sorted(SCALE_TARGETS_LIST[idx], reverse=True))

        scores = [compute_score(scales_per_symbol[s]) for s in symbols]
        scale_targets[np.argsort(scores)] = scale_targets

        for s, scale in zip(symbols, scale_targets):
            scales_per_symbol[s].append(scale)

        all_scale_targets.append(scale_targets.tolist())

    return all_scale_targets


def generate_card(out_card_path: str,
                  masks_folder: str,
                  symbols_folder: str,
                  card_size_pix: int,
                  circle_width_pix: int,
                  n_iter: int,
                  names: List[str],
                  symbols: List[int],
                  scale_targets: List[float]):
    """Draw symbols on a card according to the scale targets and save it as an image."""
    masks = [cv2.imread(os.path.join(masks_folder, names[symbol_idx]),
                        cv2.IMREAD_GRAYSCALE)
             for symbol_idx in symbols]

    card = Card(masks, scale_targets)
    for _ in range(n_iter):
        card.next()

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

        cropped_card_img, cropped_resized_symbol = get_overlapping_image_ranges(card_img, resized_symbol,
                                                                                x_left=x_left, y_top=y_top)
        _, cropped_resized_mask = get_overlapping_image_ranges(card_img, resized_mask > 0,
                                                               x_left=x_left, y_top=y_top)
        assert cropped_card_img.shape[:2] == cropped_resized_mask.shape

        cropped_card_img[cropped_resized_mask] = cropped_resized_symbol[cropped_resized_mask]

    if circle_width_pix is not None:
        cv2.circle(card_img, (card_size_pix // 2, card_size_pix // 2),
                   card_size_pix // 2, (0, 0, 0), circle_width_pix)
    cv2.imwrite(out_card_path, card_img)

    if DEBUG_FINAL:
        cv2.imshow("Card Image", card_img)
        cv2.waitKey(1)


@profile
def main(masks_folder: str,
         symbols_folder: str,
         out_cards_folder: str,
         card_size_pix: int,
         circle_width_pix: Optional[int],
         n_iter: int):
    """Generate 57 Dobble cards from symbols masks and images.

    Args:
        masks_folder: Folder containing the low-resolution symbols masks images
        symbols_folder: Folder containig the high-resolution symbols colored images
        out_cards_folder: Output folder containing the high-resolution random drawn cards
        card_size_pix: Size of the output high-resolution cards
        circle_width_pix: Width of the circle around each card. Use None to remove circle. Covariant with card_size_pix
        n_iter: Number of evolution steps for each card
    """
    names = list_image_files(masks_folder)
    assert_len(names, 57)

    cards = get_cards()
    assert_len(cards, 57)

    new_folder(out_cards_folder)

    scale_targets_per_card = allocate_scale_targets(cards)
    assert_len(scale_targets_per_card, 57)

    list_kwargs = [{"out_card_path": os.path.join(out_cards_folder, f"card_{card_idx}.png"),
                    "masks_folder": masks_folder,
                    "symbols_folder": symbols_folder,
                    "card_size_pix": card_size_pix,
                    "circle_width_pix": circle_width_pix,
                    "n_iter": n_iter,
                    "names": names,
                    "symbols": symbols,
                    "scale_targets": scale_targets}
                   for card_idx, (symbols, scale_targets) in enumerate(zip(cards, scale_targets_per_card))]

    multiprocess(generate_card, list_kwargs, tqdm_title="Generate Cards",
                 n_jobs=1 if DEBUG or DEBUG_FINAL else -1)
