# /usr/bin/python3
"""Dobble"""
import os

import fire

from dobble import card
from dobble import mask
from dobble import pdf
from dobble import preprocess
from dobble.utils import new_folder


def main(symbols_folder: str,
         output_folder: str,
         mask_computing_size_pix: int = 300,
         mask_low_res_size_pix: int = 100,
         mask_margin_pix: int = 12,
         mask_ths: int = 250,
         card_size_pix: int = 3000,
         circle_width_pix: int = 10,
         card_n_iter: int = 1500,
         card_size_cm: float = 13.):
    """Generate Dobble PDF from 57 symbol images.

    Args:
        symbols_folder: Input Folder containing the 57 symbol images (on a white background)
        output_folder: Output result folder
        mask_computing_size_pix: Size of the images when finding mask contours and applying dilation
        mask_low_res_size_pix: Output size of the low resolution dumped masks
        mask_margin_pix: Dilation applied around the mask, covariant with computing_size_pix
        mask_ths: Pixels the intensity of which is above this threshold are considered as white background
        card_size_pix: Size of the output high-resolution cards
        circle_width_pix: Width of the circle around each card. Covariant with card_size_pix
        card_n_iter: Number of evolution steps for each card
        card_size_cm: Diameter of the output Dobble cards to print
    """
    new_folder(output_folder)

    square_symbols_folder = os.path.join(output_folder, "1_square_symbols")
    masks_folder = os.path.join(output_folder, "2_masks")
    cards_folder = os.path.join(output_folder, "3_cards")
    print_folder = os.path.join(output_folder, "4_print")

    preprocess.main(images_folder=symbols_folder,
                    out_images_folder=square_symbols_folder)

    mask.main(symbols_folder=square_symbols_folder,
              out_masks_folder=masks_folder,
              computing_size_pix=mask_computing_size_pix,
              low_res_size_pix=mask_low_res_size_pix,
              margin_pix=mask_margin_pix,
              ths=mask_ths)

    card.main(masks_folder=masks_folder,
              symbols_folder=square_symbols_folder,
              out_cards_folder=cards_folder,
              card_size_pix=card_size_pix,
              circle_width_pix=circle_width_pix,
              n_iter=card_n_iter)

    pdf.main(cards_folder=cards_folder,
             out_print_folder=print_folder,
             card_size_cm=card_size_cm)


if __name__ == "__main__":
    fire.Fire(main)
