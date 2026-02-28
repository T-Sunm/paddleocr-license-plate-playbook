#!/usr/bin/env python3
"""
Recognition Inference for PaddleOCR.
Supports: Single Image and Batch (GT file) modes.
"""
import argparse
import subprocess
import os
import sys
import csv
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from _paths import PROJECT_ROOT, PADDLE_OCR_ROOT, CONFIGS_DIR, OUTPUT_DIR
from utils import norm_edit_dis, find_checkpoint, parse_rec_output, parse_batch_rec_output, extract_lr_num


def get_paddle_env() -> dict:
    """Get environment for subprocess."""
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{PADDLE_OCR_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def resolve_checkpoint(checkpoint_path: Optional[str], model_num: int) -> Optional[Path]:
    """Resolve checkpoint path to absolute path."""
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        return PROJECT_ROOT / ckpt if not ckpt.is_absolute() else ckpt
    return find_checkpoint(model_num)


def run_inference(image_path: str, model_num: int, checkpoint_path: Optional[str] = None, batch: bool = False) -> dict:
    """Run inference on single image or directory."""
    config = CONFIGS_DIR / "ensemble_v5" / "rec" / f"rec_v5_model_{model_num}.yml"
    checkpoint = resolve_checkpoint(checkpoint_path, model_num)
    
    if not config.exists() or checkpoint is None:
        return {} if batch else ("", 0.0)
    
    cmd = [
        sys.executable, "tools/infer_rec.py",
        "-c", str(config),
        "-o", f"Global.checkpoints={checkpoint}", f"Global.infer_img={image_path}"
    ]
    
    try:
        result = subprocess.run(
            cmd, cwd=str(PADDLE_OCR_ROOT), capture_output=True, text=True, env=get_paddle_env()
        )
        output = result.stdout + result.stderr
        return parse_batch_rec_output(output) if batch else parse_rec_output(output)
    except Exception:
        return {} if batch else ("", 0.0)


def run_single_mode(args) -> None:
    model_num = args.model or (((extract_lr_num(args.image) - 1) % 5) + 1 if extract_lr_num(args.image) > 0 else 1)
    text, score = run_inference(args.image, model_num, args.checkpoint)
    
    print(f"\nImage: {args.image} (Model {model_num})")
    print(f"Prediction: {text} (Score: {score:.4f})")
    if args.gt:
        print(f"GT: {args.gt} | Match: {text == args.gt} | NED: {norm_edit_dis(text, args.gt):.4f}")


def _save_csv(results: list[dict], output_csv: str) -> None:
    """Save inference results to CSV using a single DictWriter properly."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    em = sum(r['exact_match'] for r in results) / len(results)
    ned = sum(r['ned'] for r in results) / len(results)
    print(f"  ✓ Saved: {output_csv}")
    print(f"  ✓ Metrics - EM: {em:.4f}, NED: {ned:.4f}")


def run_batch_mode(args) -> None:
    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f"Error: GT file {args.gt} not found")
        return
    
    gt_data = [(p[0], p[1]) for line in open(gt_path, 'r', encoding='utf-8') 
               if len(p := line.strip().split('\t')) >= 2]
    
    model_nums = [args.model] if args.model else range(1, 6)
    results_dir = OUTPUT_DIR / "inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for m in model_nums:
        print(f"\n[Batch] Processing Model {m}...")
        
        # Filter by lr-00X pattern (skip for val sets)
        is_val = "val" in gt_path.name.lower()
        filtered = [(p, l) for p, l in gt_data if is_val or (lr := extract_lr_num(p)) == 0 or lr == m]
        
        if not filtered:
            print(f"  No samples for Model {m}")
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            alias_map = {}
            
            for img_path, _ in filtered:
                src = Path(img_path).resolve()
                if src.exists():
                    alias = f"{hashlib.md5(str(src).encode()).hexdigest()[:12]}{src.suffix}"
                    os.symlink(src, tmp_path / alias)
                    alias_map[img_path] = alias
            
            print(f"  Running inference on {len(filtered)} samples...")
            preds = run_inference(str(tmp_path), m, args.checkpoint, batch=True)
            
            results = []
            for img_path, label in filtered:
                alias = alias_map.get(img_path)
                pred, score = preds.get(alias, ("", 0.0)) if alias else ("", 0.0)
                results.append({
                    'image': img_path, 'ground_truth': label, 'prediction': pred,
                    'score': f"{score:.4f}", 'exact_match': float(pred == label),
                    'ned': norm_edit_dis(pred, label)
                })
            
            if results:
                output_csv = args.output or str(results_dir / f"rec_results_model_{m}_{timestamp}.csv")
                _save_csv(results, output_csv)


def main():
    parser = argparse.ArgumentParser(description="Recognition Inference")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--gt", type=str, help="GT string (single) or GT file path (batch)")
    parser.add_argument("--model", type=int, choices=[1, 2, 3, 4, 5], help="Model number (1-5)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--output", type=str, help="Output CSV path (batch mode)")
    args = parser.parse_args()

    if args.image:
        run_single_mode(args)
    elif args.gt:
        run_batch_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
