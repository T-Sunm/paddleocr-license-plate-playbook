"""Checkpoint finding utilities for trained PaddleOCR models."""
from pathlib import Path
from typing import Optional

from plate_ocr.paths import OUTPUT_DIR


def find_checkpoint(model_num: int, model_type: str = "rec") -> Optional[Path]:
    """
    Find the best available checkpoint for a specific model number.
    Priority: best_accuracy > best_model/model > latest
    """
    output_dir = OUTPUT_DIR / "ensemble_v5" / model_type / f"model_{model_num}"

    best_acc = output_dir / "best_accuracy"
    if best_acc.with_suffix(".pdparams").exists():
        return best_acc

    best_model = output_dir / "best_model" / "model"
    if best_model.with_suffix(".pdparams").exists():
        return best_model

    latest = output_dir / "latest"
    if latest.with_suffix(".pdparams").exists():
        return latest

    return None
