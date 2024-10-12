# /usr/bin/python3
"""Merge Dobble cards into a scaled PDF ready to print"""


import math
import os

import cv2
import img2pdf
import numpy as np
from tqdm import tqdm

from dobble.profiling import profile
from dobble.utils import assert_len
from dobble.utils import list_image_files
from dobble.utils import new_folder


@profile
def main(cards_folder: str,
         out_print_folder: str,
         card_size_cm: float):
    """
    Merge Dobble cards into a scaled PDF ready to print

    Args:
        cards_folder: Folder containing the high-res Dobble cards images
        out_print_folder: Output folder containing the batched cards and the PDF file
        card_size_cm: Diameter of the output Dobble cards to print
    """
    names = list_image_files(cards_folder)
    assert_len(names, 57)
    names += [None]  # Pad to have an even size

    pdf_path = os.path.join(out_print_folder, "cards.pdf")
    batches_folder = os.path.join(out_print_folder, "batches")
    new_folder(batches_folder)

    first_img = cv2.imread(os.path.join(cards_folder, names[0]))
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

    assert w_pad > 0 and h_pad > 0
    dh = int(h_pad/(1+nb_patch_in_h))
    h_patch = 255*np.ones((dh, w_num_pix, 3), np.uint8)
    h_patch_bot = 255*np.ones((h_pad-(dh*nb_patch_in_h), w_num_pix, 3), np.uint8)

    dw = int(w_pad/(1+nb_patch_in_w))
    w_patch = 255*np.ones((card_size_pix, dw, 3), np.uint8)
    w_patch_right = 255*np.ones((card_size_pix, w_pad-(dw*nb_patch_in_w), 3), np.uint8)

    nb_patch_per_batch = nb_patch_in_w*nb_patch_in_h
    nb_of_batch = math.ceil(len(names)/(nb_patch_per_batch))
    batches_paths = []
    for k in tqdm(range(nb_of_batch), "Batch cards"):
        batch_path = os.path.join(batches_folder, f"batch_cards_{k}.png")

        batch_images = [cv2.imread(os.path.join(cards_folder, name))
                        if name is not None else 255*np.ones_like(first_img)
                        for name in names[
                            nb_patch_per_batch*k:nb_patch_per_batch*k+nb_patch_per_batch
                            ]]

        columns = []
        for column in range(nb_patch_in_h):
            columns.append(h_patch)
            temp_line = []
            for line in range(nb_patch_in_w):
                temp_line.append(w_patch)
                idx = line*nb_patch_in_h+column
                if idx < len(batch_images):
                    temp_line.append(batch_images[idx])
                else: # if no more cards to add, pad with white
                    temp_line.append(255*np.ones_like(first_img))
            temp_line.append(w_patch_right)
            batch_img = cv2.hconcat(temp_line)
            columns.append(batch_img)

        columns.append(h_patch_bot)
        batch_img = cv2.vconcat(columns)

        cv2.imwrite(batch_path, batch_img)
        batches_paths.append(batch_path)

    a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
    layout_fun = img2pdf.get_layout_fun(a4inpt)
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(batches_paths, layout_fun=layout_fun))

    print(f"Congratulations! Your Dobble has been saved at {os.path.abspath(pdf_path)}")
