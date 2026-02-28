"""Centralized path resolution for the plate_ocr package.

This module is the single source of truth for all project paths.
Import from anywhere: `from plate_ocr.paths import PROJECT_ROOT`
"""
from pathlib import Path

# This file lives at: src/plate_ocr/paths.py
# So PROJECT_ROOT is 3 levels up: src/plate_ocr/ -> src/ -> project_root/
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

PADDLE_OCR_ROOT = PROJECT_ROOT / "PaddleOCR"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"
