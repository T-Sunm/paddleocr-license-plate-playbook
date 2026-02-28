"""
File __init__.py để làm package
"""
from .normalize import _normalize_plate, _clamp01, _to_float
from .format_rule import detect_format, format_weight
from .voting import build_candidates_with_format, prioritize_by_format
from .scoring import score_candidate, similarity_score
from .fuse import fuse_all_tracks, fuse_track_predictions
from .io_utils import read_test_results, save_submission, save_debug

__all__ = [
    "_normalize_plate",
    "_clamp01",
    "_to_float",
    "detect_format",
    "format_weight",
    "build_candidates_with_format",
    "prioritize_by_format",
    "score_candidate",
    "similarity_score",
    "fuse_all_tracks",
    "fuse_track_predictions",
    "read_test_results",
    "save_submission",
    "save_debug",
]
