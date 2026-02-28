"""Parsing utilities for PaddleOCR output text."""
import re
from pathlib import Path
from typing import Tuple, Dict


def parse_rec_output(output: str) -> Tuple[str, float]:
    """Parse single-image rec output."""
    results = parse_batch_rec_output(output)
    if results:
        return list(results.values())[0]
    return "", 0.0


def parse_batch_rec_output(output: str) -> Dict[str, Tuple[str, float]]:
    """
    Parse PaddleOCR record output for multiple images.
    Returns dict: { 'filename': (text, score) }
    """
    results = {}
    current_img = None

    for line in output.split('\n'):
        if "Predicts of" in line:
            match = re.search(r"Predicts of (.+?):", line)
            if match:
                current_img = Path(match.group(1)).name
        elif "infer_img:" in line:
            match = re.search(r"infer_img: (.+)", line)
            if match:
                current_img = Path(match.group(1).strip()).name

        if "result:" in line and current_img:
            parts = line.split("result:", 1)[1].strip().split()
            if len(parts) >= 2:
                results[current_img] = (parts[0], float(parts[1]))
                current_img = None
        elif "Predicts of" in line and current_img:
            res_part = line.split(":", 2)
            if len(res_part) >= 3:
                res_parts = res_part[2].strip().split()
                if res_parts:
                    results[current_img] = (res_parts[0], float(res_parts[1]) if len(res_parts) > 1 else 0.0)
                    current_img = None

    return results
