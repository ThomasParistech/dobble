# /usr/bin/python3
"""Dobble"""


import card
import mask
import pdf
import preprocess

symbols_folder = "data/symbols_clo"
square_symbols_folder = "data/square_symbols"
masks_folder = "data/masks"
cards_folder = "data/cards"
print_folder = "data/print"

preprocess.main(images_folder=symbols_folder,
                out_images_folder=square_symbols_folder)

mask.main(symbols_folder=square_symbols_folder,
          out_masks_folder=masks_folder,
          computing_size_pix=300,
          low_res_size_pix=100,
          margin_pix=12,
          ths=250)

card.main(masks_folder=masks_folder,
          symbols_folder=square_symbols_folder,
          out_cards_folder=cards_folder,
          card_size_pix=3000,
          circle_width_pix=10,
          n_iter=1500)

pdf.main(cards_folder=cards_folder,
         out_print_folder=print_folder,
         card_size_cm=13)
