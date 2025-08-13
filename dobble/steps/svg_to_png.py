# /usr/bin/python3
"""Convert any SVG images to PNG format."""
import glob
import os

import imagesize
from tqdm import tqdm

from dobble.utils.file import copy_file
from dobble.utils.file import list_image_files
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
    # Copy the already rasterized images to the output folder
    rasterized_image_names = list_image_files(images_folder)
    for img_name in rasterized_image_names:
        copy_file(os.path.join(images_folder, img_name),
                  os.path.join(out_images_folder, img_name))

    # Rasterize SVG images
    svg_files = glob.glob(os.path.join(images_folder, '*.svg'))
    for in_path in tqdm(svg_files, desc="SVG to PNG"):
        out_path = os.path.join(out_images_folder, os.path.basename(in_path).replace('.svg', '.png'))
        convert_svg_to_png(in_path, out_path, largest_svg_side_pix)

    logger.info(f"{len(svg_files)} SVG images have been rasterized to PNG")
