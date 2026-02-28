"""
Chấm điểm ứng cử - Cải tiến với thông tin định dạng
"""
from typing import List, Tuple
from .format_rule import format_weight


def similarity_score(s1: str, s2: str) -> float:
    """
    Tính độ tương tự giữa 2 chuỗi (normalized edit distance)
    1.0 = giống hệt, 0.0 = hoàn toàn khác
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
    diff_count += abs(len(s1) - len(s2))
    
    return max(0.0, 1.0 - (diff_count / max_len))


def score_candidate(
    candidate: str,
    candidate_format: str,
    preds: List[str],
    confs: List[float]
) -> float:
    """
    Chấm điểm một ứng cử với thông tin định dạng
    
    Công thức:
        score = 0.45×exact + 0.40×soft + 0.15×format_bonus
    
    Args:
        candidate: Ứng cử (chuỗi đã normalize)
        candidate_format: Định dạng của ứng cử ("mercosur"/"old"/"unknown")
        preds: Danh sách dự đoán đã normalize
        confs: Danh sách độ tin cậy
    """
    # 1. Exact match: tổng confidence nếu dự đoán == ứng cử
    exact_score = sum(c for p, c in zip(preds, confs) if p == candidate)
    
    # 2. Soft match: tổng (confidence × độ tương tự)
    soft_score = sum(
        c * similarity_score(p, candidate)
        for p, c in zip(preds, confs)
    )
    
    # 3. Format bonus: ưu tiên ứng cử có định dạng hợp lệ
    format_bonus = format_weight(candidate_format)
    
    # Chuẩn hóa (để tất cả thành scale 0-1)
    total_conf = sum(confs) if confs else 1.0
    if total_conf == 0:
        total_conf = 1.0
    
    exact_normalized = exact_score / total_conf
    soft_normalized = soft_score / total_conf
    
    # Kết hợp
    score = (
        0.45 * exact_normalized +
        0.40 * soft_normalized +
        0.15 * format_bonus
    )
    
    return min(1.0, score)  # Giới hạn max 1.0
