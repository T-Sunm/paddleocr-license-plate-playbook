"""
Xây dựng danh sách ứng cử + xác định định dạng sớm
"""
from typing import List, Tuple
from .normalize import _normalize_plate, _clamp01
from .format_rule import detect_format


def build_candidates_with_format(preds: List[str], confs: List[float]) -> List[Tuple[str, str]]:
    """
    Xây dựng danh sách ứng cử cùng xác định định dạng sớm
    
    Args:
        preds: Danh sách dự đoán từ 5 ảnh
        confs: Danh sách độ tin cậy tương ứng
    
    Returns:
        List[(normalized_prediction, format)]
        Ví dụ: [("KQS2528", "old"), ("ASC06", "unknown"), ...]
    """
    # Chuẩn hóa ngay từ đầu
    normalized_preds = []
    normalized_confs = []
    
    for pred, conf in zip(preds, confs):
        norm_pred = _normalize_plate(pred)
        norm_conf = _clamp01(conf)
        
        if norm_pred:  # Chỉ thêm nếu không rỗng
            normalized_preds.append(norm_pred)
            normalized_confs.append(norm_conf)
    
    if not normalized_preds:
        return []
    
    # Xây dựng ứng cử: loại bỏ trùng lặp + sắp xếp theo confidence
    candidates_dict = {}  # {pred: conf}
    for pred, conf in zip(normalized_preds, normalized_confs):
        if pred not in candidates_dict:
            candidates_dict[pred] = conf
        else:
            # Lấy confidence cao nhất
            candidates_dict[pred] = max(candidates_dict[pred], conf)
    
    # Sắp xếp theo confidence giảm dần
    sorted_candidates = sorted(
        candidates_dict.items(),
        key=lambda x: (-x[1], x[0])  # confidence giảm, nếu bằng thì theo tên
    )
    
    # Thêm định dạng vào từng ứng cử
    result = []
    for pred, conf in sorted_candidates:
        fmt = detect_format(pred)
        result.append((pred, fmt))
    
    return result


def prioritize_by_format(candidates_with_fmt: List[Tuple[str, str]]) -> List[str]:
    """
    Ưu tiên ứng cử theo định dạng hợp lệ
    
    Nếu có ứng cử hợp lệ (mercosur/old) → chỉ trả về những cái hợp lệ
    Nếu không có → trả về tất cả
    """
    valid_candidates = [pred for pred, fmt in candidates_with_fmt if fmt != "unknown"]
    
    if valid_candidates:
        return valid_candidates
    
    # Nếu không có ứng cử hợp lệ, trả về tất cả (theo định dạng)
    return [pred for pred, fmt in candidates_with_fmt]
