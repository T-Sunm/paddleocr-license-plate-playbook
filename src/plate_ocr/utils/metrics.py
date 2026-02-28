"""Evaluation metrics for plate recognition."""
import re
from pathlib import Path
from typing import Tuple, Optional

import Levenshtein


def norm_edit_dis(pred: str, gt: str) -> float:
    """Calculate Normalized Edit Distance (1.0 = perfect match)."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return 1 - (dist / max_len)


def extract_lr_num(image_path: str) -> int:
    """Extract lr number from filename like lr-001.jpg -> 1"""
    match = re.search(r'lr-(\d+)', Path(image_path).name)
    return int(match.group(1)) if match else 0
