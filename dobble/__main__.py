# /usr/bin/python3
"""Dobble"""
import os
from typing import Optional

import fire

from dobble import card
from dobble import pdf
from dobble import preprocess
from dobble.profiling import export_profiling_events
from dobble.profiling import LogScopeTime
from dobble.utils import assert_len
from dobble.utils import list_image_files
from dobble.utils import new_folder


def main(symbols_folder: str,
         output_folder: str,
         largest_svg_side_pix: int = 3000,
         mask_computing_size_pix: int = 300,
         mask_low_res_size_pix: int = 100,
         mask_margin_pix: int = 20,
         mask_ths: int = 250,
         card_size_pix: int = 3000,
         circle_width_pix: Optional[int] = 10,
         junior_size: bool = False,
         card_n_iter: int = 1000,
         card_size_cm: float = 13.):
    """Generate Dobble PDF from 57 symbol images.

    Args:
        symbols_folder: Input Folder containing the 57 symbol images (on a white background)
        output_folder: Output result folder
        largest_svg_side_pix: Size of the largest image side (in pix) when rasterizing a SVG image
        mask_computing_size_pix: Size of the images when finding mask contours and applying dilation
        mask_low_res_size_pix: Output size of the low resolution dumped masks
        mask_margin_pix: Dilation applied around the mask, covariant with computing_size_pix
        mask_ths: Pixels the intensity of which is above this threshold are considered as white background
        card_size_pix: Size of the output high-resolution cards
        circle_width_pix: Width of the circle around each card. Use None to remove circle. Covariant with card_size_pix
        junior_size: Use the junior version of the game (6 symbols per card) instead of the standard version (8 symbols per card)
        card_n_iter: Number of evolution steps for each card
        card_size_cm: Diameter of the output Dobble cards to print
    """
    assert os.path.isdir(symbols_folder), \
        f"Input symbols folder {symbols_folder} does not exist"
    assert_len(list_image_files(symbols_folder), 57,
               msg=f"Invalid number of symbols in input folder {symbols_folder}")

    new_folder(output_folder)

    square_symbols_folder = os.path.join(output_folder, "1_square_symbols")
    masks_folder = os.path.join(output_folder, "2_masks")
    cards_folder = os.path.join(output_folder, "3_cards")
    print_folder = os.path.join(output_folder, "4_print")

    n_symbols_per_card = 6 if junior_size else 8

    with LogScopeTime("Preprocessing"):
        preprocess.main(images_folder=symbols_folder,
                        out_images_folder=square_symbols_folder,
                        out_masks_folder=masks_folder,
                        mask_computing_size_pix=mask_computing_size_pix,
                        mask_low_res_size_pix=mask_low_res_size_pix,
                        mask_margin_pix=mask_margin_pix,
                        mask_ths=mask_ths,
                        largest_svg_side_pix=largest_svg_side_pix)

    with LogScopeTime("Cards"):
        card.main(masks_folder=masks_folder,
                  symbols_folder=square_symbols_folder,
                  out_cards_folder=cards_folder,
                  card_size_pix=card_size_pix,
                  circle_width_pix=circle_width_pix,
                  n_symbols=n_symbols_per_card,
                  n_iter=card_n_iter)

    with LogScopeTime("PDF"):
        pdf.main(cards_folder=cards_folder,
                 out_print_folder=print_folder,
                 card_size_cm=card_size_cm,
                 n_symbols_per_card = n_symbols_per_card)

    export_profiling_events("data/profiling.json")


if __name__ == "__main__":
    fire.Fire(main)
