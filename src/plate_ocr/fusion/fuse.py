"""
Main Pipeline - Fuses predictions for each track
Improvement: Determine format early + use format to filter candidates
"""
import pandas as pd
from .normalize import _normalize_plate, _clamp01, _to_float
from .voting import build_candidates_with_format, prioritize_by_format
from .scoring import score_candidate


def fuse_track_predictions(
    track_group: pd.DataFrame,
    pred_col: str = "prediction",
    conf_col: str = "confidence"
) -> tuple[str, float, dict]:
    """
    Fuse predictions of 1 track (5 images)
    
    Returns:
        (best_prediction, best_score, debug_info)
    """
    track_id = track_group["track_id"].iloc[0]
    
    # Extract prediction and confidence from 5 images
    preds = [str(x) for x in track_group[pred_col].values]
    confs = [_clamp01(_to_float(x)) for x in track_group[conf_col].values]
    
    # Normalize early
    normalized_preds = [_normalize_plate(p) for p in preds]
    
    # ⭐ Xây dựng ứng cử + XÁC ĐỊNH FORMAT SỚM
    candidates_with_fmt = build_candidates_with_format(normalized_preds, confs)
    
    if not candidates_with_fmt:
        return "", 0.0, {
            "track_id": track_id,
            "result": "empty",
            "reason": "Tất cả dự đoán đều rỗng"
        }
    
    # ⭐ Lọc ứng cử ưu tiên format hợp lệ
    prioritized_candidates = prioritize_by_format(candidates_with_fmt)
    
    # Chấm điểm từng ứng cử
    best_pred = ""
    best_score = -1.0
    scores_debug = {}
    
    for pred, fmt in candidates_with_fmt:
        score = score_candidate(
            pred, fmt,
            normalized_preds, confs
        )
        scores_debug[pred] = {
            "format": fmt,
            "score": score
        }
        
        # Ưu tiên ứng cử từ danh sách prioritized
        is_prioritized = pred in prioritized_candidates
        adjusted_score = score + (0.1 if is_prioritized else 0.0)
        
        if adjusted_score > best_score:
            best_score = score  # Lưu score gốc
            best_pred = pred
    
    return best_pred, best_score, {
        "track_id": track_id,
        "candidates_count": len(candidates_with_fmt),
        "prioritized_count": len(prioritized_candidates),
        "best_prediction": best_pred,
        "best_score": best_score,
        "scores_detail": scores_debug
    }


def fuse_all_tracks(
    df: pd.DataFrame,
    track_col: str = "track_id",
    pred_col: str = "prediction",
    conf_col: str = "confidence"
) -> pd.DataFrame:
    """
    Hợp nhất dự đoán cho tất cả các track
    
    Returns:
        DataFrame với cột: track_id, prediction, confidence, format
    """
    results = []
    debug_info = []
    
    for track_id, group in df.groupby(track_col):
        best_pred, best_score, debug = fuse_track_predictions(
            group, pred_col, conf_col
        )
        
        results.append({
            "track_id": track_id,
            "prediction": best_pred,
            "confidence": best_score
        })
        
        debug_info.append(debug)
    
    result_df = pd.DataFrame(results)
    
    return result_df, debug_info
