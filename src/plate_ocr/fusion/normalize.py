import re
import pandas as pd


def _to_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _normalize_plate(s: str) -> str:
    """Chuẩn hóa biển số: xóa khoảng trắng, ký tự đặc biệt"""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _clamp01(x: float) -> float:
    """Giới hạn giá trị trong range [0.0, 1.0]"""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
