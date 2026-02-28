#!/usr/bin/env python3
"""
Export detection/recognition models to inference format.
Automatically finds best_accuracy or latest checkpoint for each model.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from plate_ocr.paths import PROJECT_ROOT, PADDLE_OCR_ROOT, CONFIGS_DIR, OUTPUT_DIR
from plate_ocr.utils.checkpoint import find_checkpoint


class ModelExporter:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.paddle_root = PADDLE_OCR_ROOT
        self.config_dir = CONFIGS_DIR / "ensemble" / model_type
        self.output_dir = OUTPUT_DIR / "ensemble_v5" / model_type

    def export_model(self, model_num: int) -> bool:
        """Export a single model to inference format."""
        config_file = self.config_dir / f"{self.model_type}_v5_model_{model_num}.yml"
        checkpoint = find_checkpoint(model_num, self.model_type)

        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            return False

        if checkpoint is None:
            return False

        inference_dir = self.output_dir / f"model_{model_num}" / "inference"

        python_exe = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python_exe.exists():
            print(f"Error: Virtual environment not found. Please run 'uv sync' first.")
            return False

        cmd = [
            str(python_exe),
            "tools/export_model.py",
            "-c", str(config_file),
            "-o",
            f"Global.pretrained_model={checkpoint}",
            f"Global.save_inference_dir={inference_dir}",
            "Global.export_with_pir=True"
        ]

        print(f"\nExporting {self.model_type} model_{model_num}...")
        print(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["FLAGS_enable_pir_api"] = "1"
        env["FLAGS_enable_pir_in_executor"] = "1"

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.paddle_root),
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            print(f"Successfully exported model_{model_num} to {inference_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to export model_{model_num}")
            print(f"Error: {e.stderr}")
            return False

    def export_models(self, model_nums: List[int]):
        """Export specified models."""
        print(f"Starting {self.model_type} models export...")
        print(f"PaddleOCR root: {self.paddle_root}")
        print(f"Models to export: {model_nums}")

        success_count = 0
        for model_num in model_nums:
            if self.export_model(model_num):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Export completed: {success_count}/{len(model_nums)} models exported successfully")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PaddleOCR ensemble models to inference format"
    )
    parser.add_argument(
        "--type",
        choices=["det", "rec"],
        default="det",
        help="Model type to export (default: det)"
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Specific model number to export (1-5). If not specified, exports all 5 models"
    )

    args = parser.parse_args()

    print(f"Detected workspace root: {PROJECT_ROOT}")
    exporter = ModelExporter(args.type)

    model_nums = [args.model] if args.model else list(range(1, 6))
    exporter.export_models(model_nums)


if __name__ == "__main__":
    main()
