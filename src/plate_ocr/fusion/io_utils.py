"""
Tiện ích I/O - Đọc CSV và xuất kết quả
"""
import pandas as pd
import os


def read_test_results(csv_path: str) -> pd.DataFrame:
    """Đọc file test_results.csv"""
    df = pd.read_csv(csv_path)
    return df


def save_submission(
    df: pd.DataFrame,
    output_path: str,
    pred_col: str = "prediction",
    conf_col: str = "confidence"
):
    """
    Lưu kết quả dưới định dạng:
    track_id,prediction;confidence
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            track_id = row["track_id"]
            pred = row[pred_col]
            conf = row[conf_col]
            
            if conf > 0:
                line = f"{track_id},{pred};{conf:.4f}"
            else:
                line = f"{track_id},;"
            
            f.write(line + "\n")


def save_debug(
    debug_info: list,
    output_path: str
):
    """Lưu thông tin debug chi tiết"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    debug_df = pd.json_normalize(debug_info)
    debug_df.to_csv(output_path, index=False)
