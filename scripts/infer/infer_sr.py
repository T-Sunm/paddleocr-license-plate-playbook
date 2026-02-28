#!/usr/bin/env python3
"""
Super Resolution Inference Script.
Wrapper around PaddleOCR's infer_sr.py specifying direct checkpoint weights.
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path

try:
    from _paths import PROJECT_ROOT, PADDLE_OCR_ROOT, OUTPUT_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PADDLE_OCR_ROOT = PROJECT_ROOT / "PaddleOCR"
    OUTPUT_DIR = PROJECT_ROOT / "output"

def get_paddle_env() -> dict:
    """Get environment for subprocess."""
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{PADDLE_OCR_ROOT}:{env.get('PYTHONPATH', '')}"
    return env

def main():
    parser = argparse.ArgumentParser(description="Super Resolution Inference (Checkpoint Mode)")
    
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Đường dẫn đến ảnh hoặc thư mục chứa ảnh đầu vào")
    parser.add_argument("--config", type=str, default="configs/sr/sr_gestalt_plate.yml", 
                        help="Đường dẫn config yml (chứa SRResize imgH, imgW)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Đường dẫn file best weights checkpoint (không bao gồm đuôi .pdparams)")
    parser.add_argument("--output_dir", type=str, default="output/inference/sr_result", 
                        help="Thư mục để lưu ảnh kết quả (ảnh HR)")
    
    args = parser.parse_args()

    image_path = str(Path(args.image_dir).resolve())
    conf_path = str(Path(args.config).resolve())
    ckpt_path = str(Path(args.checkpoint).resolve())
    out_path = str(Path(args.output_dir).resolve())
    
    # Create output directory
    os.makedirs(out_path, exist_ok=True)

    print(f"[Inference SR] Running with weights: {ckpt_path}.pdparams")
    print(f"               Input: {image_path}")
    print(f"               Output: {out_path}")
    
    cmd = [
        sys.executable, "tools/infer_sr.py",
        "-c", conf_path,
        "-o",
        f"Global.checkpoints={ckpt_path}",
        f"Global.infer_img={image_path}",
        f"Global.save_visual={out_path}"
    ]
    
    subprocess.run(cmd, cwd=str(PADDLE_OCR_ROOT), env=get_paddle_env())

if __name__ == "__main__":
    main()
