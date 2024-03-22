# /usr/bin/python3
"""Dobble"""


import os
import shutil


def new_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
