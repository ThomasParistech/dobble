#!/usr/bin/python3
"""Paths."""
import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(MAIN_DIR, 'data')
ASSETS_DIR = os.path.join(MAIN_DIR, 'assets')

TEST_DIR = os.path.join(DATA_DIR, 'test')
