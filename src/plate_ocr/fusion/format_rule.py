import re

# Định dạng biển số
RE_MERCOSUR = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")  # VD: KQS2A28
RE_OLD = re.compile(r"^[A-Z]{3}[0-9]{4}$")  # VD: KQS2528


def detect_format(plate_normalized: str) -> str:
    """
    Xác định định dạng biển số
    
    Returns:
        "mercosur": Định dạng Mercosur (KQS2A28)
        "old": Định dạng cũ Brazilian (KQS2528)
        "unknown": Không xác định được
    """
    if not plate_normalized:
        return "unknown"
    
    if RE_MERCOSUR.match(plate_normalized):
        return "mercosur"
    if RE_OLD.match(plate_normalized):
        return "old"
    return "unknown"


def format_weight(fmt: str) -> float:
    """Trọng số định dạng để ưu tiên ứng cử hợp lệ"""
    if fmt == "mercosur":
        return 1.0
    if fmt == "old":
        return 0.9
    return 0.0
