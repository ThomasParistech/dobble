# /usr/bin/python3
"""Merge Dobble cards into a scaled PDF ready to print."""
import math
import os

import cv2
import img2pdf
import numpy as np
from tqdm import tqdm

from dobble.utils.asserts import assert_gt
from dobble.utils.asserts import assert_len
from dobble.utils.file import create_new_folder
from dobble.utils.file import list_image_files
from dobble.utils.image_loader import load_image
from dobble.utils.image_loader import write_image
from dobble.utils.logger import logger
from dobble.utils.np_types import NpIntArrayType
from dobble.utils.profiling import profile


def save_batch_image(out_batches_folder: str,
                     cards_folder: str,
                     names: list[str | None],
                     first_img: NpIntArrayType,
                     h_patch: NpIntArrayType,
                     h_patch_bot: NpIntArrayType,
                     w_patch: NpIntArrayType,
                     w_patch_right: NpIntArrayType,
                     nb_patch_per_batch: int,
                     nb_patch_in_h: int,
                     nb_patch_in_w: int,
                     k: int) -> str:
    """Save a batch of cards into a single image."""
    batch_path = os.path.join(out_batches_folder, f"batch_cards_{k}.png")
    batch_images = [load_image(os.path.join(cards_folder, name))
                    if name is not None else 255*np.ones_like(first_img)
                    for name in names[nb_patch_per_batch*k:nb_patch_per_batch*k+nb_patch_per_batch]]

    columns: list[cv2.typing.MatLike] = []
    for column in range(nb_patch_in_h):
        columns.append(h_patch)
        temp_line = []
        for line in range(nb_patch_in_w):
            temp_line.append(w_patch)
            idx = line*nb_patch_in_h+column
            if idx < len(batch_images):
                temp_line.append(batch_images[idx])
            else:  # if no more cards to add, pad with white
                temp_line.append(255*np.ones_like(first_img))
        temp_line.append(w_patch_right)
        batch_img = cv2.hconcat(temp_line)
        columns.append(batch_img)

    columns.append(h_patch_bot)
    batch_img = cv2.vconcat(columns)

    write_image(batch_path, batch_img)
    return batch_path


@profile
def main(cards_folder: str,
         out_print_folder: str,
         card_size_cm: float,
         n_symbols_per_card: int) -> None:
    """Dobble cards into a scaled PDF ready to print.

    Args:
        cards_folder: Folder containing the high-res Dobble cards images
        out_print_folder: Output folder containing the batched cards and the PDF file
        card_size_cm: Diameter of the output Dobble cards to print
        n_symbols_per_card: Number of symbols per card
    """
    names = list_image_files(cards_folder)

    n_cards = n_symbols_per_card**2 - n_symbols_per_card + 1
    assert_len(names, n_cards)

    pdf_path = os.path.join(out_print_folder, "cards.pdf")
    batches_folder = os.path.join(out_print_folder, "batches")
    create_new_folder(batches_folder)

    first_img = load_image(os.path.join(cards_folder, names[0]))
    card_size_pix = first_img.shape[0]
    pix_per_cm = float(card_size_pix) / card_size_cm
    w_a4_cm = 21.0
    h_a4_cm = 29.7

    w_num_pix = math.floor(w_a4_cm * pix_per_cm)
    h_num_pix = math.floor(h_a4_cm * pix_per_cm)

    # get maximum number of patch to add in the final image
    nb_patch_in_w = math.floor(w_num_pix / card_size_pix)
    nb_patch_in_h = math.floor(h_num_pix / card_size_pix)

    w_pad = w_num_pix-card_size_pix * nb_patch_in_w
    h_pad = h_num_pix-card_size_pix * nb_patch_in_h

    assert_gt(w_pad, 0)
    assert_gt(h_pad, 0)
    dh = int(h_pad/(1+nb_patch_in_h))
    h_patch = 255*np.ones((dh, w_num_pix, 3), np.uint8)
    h_patch_bot = 255*np.ones((h_pad-(dh*nb_patch_in_h), w_num_pix, 3), np.uint8)

    dw = int(w_pad/(1+nb_patch_in_w))
    w_patch = 255*np.ones((card_size_pix, dw, 3), np.uint8)
    w_patch_right = 255*np.ones((card_size_pix, w_pad-(dw*nb_patch_in_w), 3), np.uint8)

    nb_patch_per_batch = nb_patch_in_w*nb_patch_in_h
    nb_of_batch = math.ceil(len(names)/(nb_patch_per_batch))
    batches_paths = [save_batch_image(batches_folder,
                                      cards_folder,
                                      names+[None],  # Pad to have an even size
                                      first_img,
                                      h_patch,
                                      h_patch_bot,
                                      w_patch,
                                      w_patch_right,
                                      nb_patch_per_batch,
                                      nb_patch_in_h,
                                      nb_patch_in_w,
                                      k)
                     for k in tqdm(range(nb_of_batch), "Batch cards")]

    a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
    layout_fun = img2pdf.get_layout_fun(a4inpt)

    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(batches_paths, layout_fun=layout_fun))

    logger.info(f"Congratulations! Your Dobble has been saved at {os.path.abspath(pdf_path)}")
