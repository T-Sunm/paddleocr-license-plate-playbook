"""
Prepare PaddleOCR Ensemble Recognition Data.
Generates 5 distinct datasets (folds/variations) for ensemble recognition models.

Usage:
    python scripts/data/prepare_ensemble.py
"""
import argparse
import logging
import sys
import shutil
import random
import json
from pathlib import Path
from typing import List, Optional

from plate_ocr.paths import PROJECT_ROOT
# Adjust internal imports to support current package structure
from plate_ocr.data.track_loader import TrackLoader, TrackInfo
from plate_ocr.data.sample import Sample
from plate_ocr.utils.file_utils import ensure_dir

# Configuration
DATA_MODELS_DIR = PROJECT_ROOT / "data/models/paddleocr"
PROCESSED_REC_DIR = PROJECT_ROOT / "data/processed_rec"
NUM_MODELS = 5
SEED_SPLIT = 42


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("prepare_ensemble")


def clean_output_dir(directory: Path):
    if directory.exists():
        logging.info(f"Cleaning existing directory: {directory}")
        shutil.rmtree(directory)
    ensure_dir(directory)


def is_sample_for_model(file_path: Path, model_idx: int) -> bool:
    """Check if sample belongs to specific model based on index suffix (-001 to -005)."""
    return f"-00{model_idx}" in file_path.name


def save_rec_split(input_label: Path, output_dir: Path, label_filename: str, 
                   images_subdir: str, prefix: str, logger, 
                   model_idx: int = None, filter_by_index: bool = False):
    """Save recognition dataset split with text labels."""
    images_dir = output_dir / images_subdir
    ensure_dir(images_dir)
    label_file = output_dir / label_filename
    
    count = 0
    with open(input_label, 'r', encoding='utf-8') as f_in, \
         open(label_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            src_path_str, text = parts
            
            # Normalize path
            normalized = src_path_str.replace("\\", "/")
            if "plate-recognition-low-resolution/" in normalized:
                rel_path = normalized.split("plate-recognition-low-resolution/")[-1]
                src_path = PROJECT_ROOT / rel_path
            else:
                src_path = Path(src_path_str)
            
            if not src_path.exists():
                continue
            
            # Filter for model if needed
            if filter_by_index and model_idx:
                if not is_sample_for_model(src_path, model_idx):
                    continue
            
            # Create filename
            if "cropped" in src_path.name:
                base_name = src_path.name
            else:
                # Fallback for raw tracks if needed
                try:
                    country = src_path.parent.parent.name
                    track = src_path.parent.name
                    base_name = f"{country}_{track}_{src_path.name}"
                except:
                    base_name = src_path.name
            
            new_filename = f"{prefix}_{count:06d}_{base_name.replace(' ', '_').lower()}"
            dst_path = images_dir / new_filename
            
            try:
                shutil.copy2(src_path, dst_path)
                f_out.write(f"{images_subdir}/{new_filename}\t{text}\n")
                count += 1
            except Exception as e:
                logger.error(f"Copy failed: {e}")
    
    logger.info(f"  Saved {count} samples to {label_filename}")
    return count


def create_plate_dict(output_dir: Path):
    """Create character dictionary for license plates."""
    dict_file = output_dir / "plate_dict.txt"
    chars = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["-"]
    with open(dict_file, 'w', encoding='utf-8') as f:
        for c in chars:
            f.write(f"{c}\n")
    return dict_file


def prepare_recognition_ensemble(logger):
    """Prepare recognition ensemble data."""
    output_base = DATA_MODELS_DIR / "rec_data"
    clean_output_dir(output_base)
    
    train_label = PROCESSED_REC_DIR / "train_rec.txt"
    val_label = PROCESSED_REC_DIR / "val_rec.txt"
    test_label = PROCESSED_REC_DIR / "test_rec.txt"
    
    if not train_label.exists():
        logger.error(f"Train label not found: {train_label}")
        logger.error("Run 'python scripts/data/build_dataset.py --mode rec' first.")
        return
    
    logger.info(f"Generating recognition data for {NUM_MODELS} models...")
    
    for i in range(NUM_MODELS):
        model_idx = i + 1
        model_dir = output_base / f"model_{model_idx}"
        ensure_dir(model_dir)
        logger.info(f"--- Recognition Model {model_idx} ---")
        
        # Train (filtered by index)
        save_rec_split(train_label, model_dir, "train.txt", "train", "train", 
                      logger, model_idx=model_idx, filter_by_index=True)
        
        # Val (full set)
        save_rec_split(val_label, model_dir, "val.txt", "val", "val", logger)
        
        # Test (full set) - optional
        if test_label.exists():
            save_rec_split(test_label, model_dir, "test.txt", "test", "test", logger)
            
    # Create shared dict
    dict_dir = PROJECT_ROOT / "data/dict"
    ensure_dir(dict_dir)
    create_plate_dict(dict_dir)
    
    logger.info(f"Recognition ensemble data saved to: {output_base}")


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("PaddleOCR Ensemble Recognition Data Preparation")
    logger.info("=" * 60)
    
    prepare_recognition_ensemble(logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Ensemble recognition data preparation complete!")
    logger.info(f"Output: {DATA_MODELS_DIR / 'rec_data'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
