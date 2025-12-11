import cv2
import numpy as np


def draw_shape_border(img: np.ndarray, card_size_pix: int,
                      thickness: int, hexagon: bool) -> None:
    """Draw a circle or hexagon border on an image."""
    center = (card_size_pix // 2, card_size_pix // 2)
    radius = card_size_pix // 2
    if hexagon:
        hexagon_pts = _get_hexagon_points(center, radius)
        cv2.polylines(img, [hexagon_pts], isClosed=True,
                      color=(0, 0, 0), thickness=thickness)
    else:
        cv2.circle(img, center, radius, (0, 0, 0), thickness)


def draw_shape_mask(mask: np.ndarray, center: tuple[int, int], hexagon: bool) -> None:
    """Draw the card shape (circle or hexagon) on a mask."""
    if hexagon:
        hexagon_pts = _get_hexagon_points(center, center[0])
        cv2.fillPoly(mask, [hexagon_pts], color=0)
    else:
        cv2.circle(mask, center, center[0], 0, -1)


def _get_hexagon_points(center: tuple[int, int], radius: int) -> np.ndarray:
    """Calculate hexagon vertices."""
    center_x, center_y = center
    angles = np.linspace(0, 2*np.pi, 7)
    hexagon_pts = np.array([[center_x + int(radius * np.cos(angle)),
                             center_y + int(radius * np.sin(angle))]
                            for angle in angles], np.int32)
    return hexagon_pts.reshape((-1, 1, 2))
