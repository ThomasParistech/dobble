# /usr/bin/python3
"""Convert any SVG images to PNG format."""
import os

import imagesize
from tqdm import tqdm

from dobble.utils.file import copy_file
from dobble.utils.file import create_new_folder
from dobble.utils.file import list_image_files
from dobble.utils.file import list_svg_files
from dobble.utils.logger import logger
from dobble.utils.profiling import profile


def _convert_svg_to_png(in_path: str, out_path: str, largest_side_pix: int, set_width: bool) -> None:
    option = "output-width" if set_width else "output-height"
    cmd = f"cairosvg '{in_path}' -o '{out_path}' --{option} {largest_side_pix}"
    os.system(cmd)


def convert_svg_to_png(in_path: str, out_path: str, largest_side_pix: int) -> None:
    """Convert SVG to PNG using cairosvg."""
    _convert_svg_to_png(in_path, out_path, largest_side_pix, set_width=True)
    width, height = imagesize.get(out_path)
    if height > width:
        _convert_svg_to_png(in_path, out_path, largest_side_pix, set_width=False)


@profile
def main(images_folder: str,
         out_images_folder: str,
         largest_svg_side_pix: int) -> None:
    """Convert any SVG images to PNG format.

    Args:
        images_folder: Input folder containing colored images either rasterized or vectorized (SVG)
        out_images_folder: Output folder containing the rasterized images
        largest_svg_side_pix: Size of the largest image side (in pix) when rasterizing a SVG image
    """
    create_new_folder(out_images_folder)

    # Copy the already rasterized images to the output folder
    rasterized_image_names = list_image_files(images_folder)
    for img_name in rasterized_image_names:
        copy_file(os.path.join(images_folder, img_name),
                  os.path.join(out_images_folder, img_name))

    # Rasterize SVG images
    svg_names = list_svg_files(images_folder)
    for svg_name in tqdm(svg_names, desc="SVG to PNG"):
        convert_svg_to_png(os.path.join(images_folder, svg_name),
                           os.path.join(out_images_folder, svg_name.replace('.svg', '.png')),
                           largest_svg_side_pix)

    logger.info(f"{len(svg_names)} SVG images have been rasterized to PNG")
