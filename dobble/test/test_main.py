#!/usr/bin/python3
"""Test Main."""
import os

from dobble.__main__ import main
from dobble.utils.asserts import assert_isfile
from dobble.utils.file import create_new_folder
from dobble.utils.paths import ASSETS_DIR
from dobble.utils.paths import TEST_DIR


def test_normal_main() -> None:
    """Test normal main."""
    # GIVEN
    symbols_folder = os.path.join(ASSETS_DIR, 'symbols_examples')

    # WHEN
    output_folder = os.path.join(TEST_DIR, 'result_normal')
    main(symbols_folder=symbols_folder, output_folder=output_folder)

    # THEN
    assert_isfile(os.path.join(output_folder, '4_print', "cards.pdf"))


def test_junior_main() -> None:
    """Test junior main."""
    # GIVEN
    symbols_folder = os.path.join(ASSETS_DIR, 'symbols_examples_junior')

    # WHEN
    output_folder = os.path.join(TEST_DIR, 'result_junior')
    main(symbols_folder=symbols_folder, output_folder=output_folder, junior_size=True)

    # THEN
    assert_isfile(os.path.join(output_folder, '4_print', "cards.pdf"))


def test_junior_main_with_svg() -> None:
    """Test junior main but with only SVG images."""
    # GIVEN
    output_folder = os.path.join(TEST_DIR, 'result_junior_svg')
    symbols_folder = os.path.join(TEST_DIR, 'svg_junior_symbols')
    create_new_folder(symbols_folder)

    for k in range(31):
        path = os.path.join(symbols_folder, f"symbol_{k}.svg")
        color = 'red' if k % 2 == 0 else 'blue'
        with open(path, 'w') as f:
            f.write(f"""
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="white"/>
<text x="50%" y="50%" font-size="100" text-anchor="middle" dominant-baseline="central" fill="{color}">
{k}
</text>
</svg>
""")

    # WHEN
    main(symbols_folder=symbols_folder, output_folder=output_folder, junior_size=True)

    # THEN
    assert_isfile(os.path.join(output_folder, '4_print', "cards.pdf"))
