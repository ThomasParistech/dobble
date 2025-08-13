#!/usr/bin/python3
"""Test Main."""
import os

from dobble.__main__ import main
from dobble.utils.asserts import assert_isfile
from dobble.utils.paths import ASSETS_DIR
from dobble.utils.paths import TEST_DIR


def test_normal_main() -> None:
    """Test normal main."""
    output_folder = os.path.join(TEST_DIR, 'result_normal')

    main(symbols_folder=os.path.join(ASSETS_DIR, 'symbols_examples'),
         output_folder=output_folder)

    assert_isfile(os.path.join(output_folder, '4_print', "cards.pdf"))


def test_junior_main() -> None:
    """Test junior main."""
    output_folder = os.path.join(TEST_DIR, 'result_junior')

    main(symbols_folder=os.path.join(ASSETS_DIR, 'symbols_examples_junior'),
         output_folder=output_folder,
         junior_size=True)

    assert_isfile(os.path.join(output_folder, '4_print', "cards.pdf"))
