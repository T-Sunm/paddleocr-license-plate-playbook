#!/usr/bin/env python3
"""
Evaluate recognition models using PaddleOCR tools/eval.py.
Uses checkpoints (.pdparams) directly.
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path
from _paths import PROJECT_ROOT
from utils import find_checkpoint

def evaluate_model(model_num: int, model_type: str = "rec"):
    workspace_root = PROJECT_ROOT
    paddle_root = workspace_root / "approaches" / "paddle"
    config_path = paddle_root / "configs" / "ensemble_v5" / model_type / f"{model_type}_v5_model_{model_num}.yml"
    checkpoint = find_checkpoint(model_num, model_type)
    
    if not config_path.exists() or checkpoint is None:
        print(f"Error: Missing config or checkpoint for model {model_num}")
        return

    python_exe = paddle_root / ".venv" / "bin" / "python"
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{str(paddle_root / 'PaddleOCR')}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        str(python_exe), "tools/eval.py",
        "-c", str(config_path),
        "-o", f"Global.checkpoints={checkpoint}"
    ]
    
    print(f"Evaluating {model_type} model {model_num}...")
    subprocess.run(cmd, cwd=str(paddle_root / "PaddleOCR"), env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, required=True, choices=[1,2,3,4,5])
    parser.add_argument("--type", type=str, choices=["rec", "det"], default="rec")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.type)
