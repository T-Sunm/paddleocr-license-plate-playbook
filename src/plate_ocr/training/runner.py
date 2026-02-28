"""Base training runner — shared logic for all train_*.py scripts."""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from plate_ocr.paths import PADDLE_OCR_ROOT, OUTPUT_DIR


def build_env() -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{PADDLE_OCR_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def run_paddle_training(
    config_path: Path,
    task_name: str,
    extra_overrides: Optional[list[str]] = None,
    seed: int = 42,
) -> None:
    """Run PaddleOCR training with standard error handling."""
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        sys.exit(1)

    output_dir = OUTPUT_DIR / f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    cmd = [
        sys.executable, "tools/train.py",
        "-c", str(config_path),
        "-o", f"Global.seed={seed}",
        "-o", f"Global.save_model_dir={output_dir}",
    ]
    if extra_overrides:
        for override in extra_overrides:
            cmd += ["-o", override]

    print(f"\n{'='*80}\n🚀 STARTING {task_name.upper()}\n   Config: {config_path}\n   Output: {output_dir}\n{'='*80}\n")

    try:
        subprocess.run(cmd, cwd=str(PADDLE_OCR_ROOT), env=build_env(), check=True)
        print(f"\n[✓] Finished {task_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] {task_name} failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(1)
