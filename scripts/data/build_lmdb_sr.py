#!/usr/bin/env python3
"""
Build LMDB dataset for SR (Super Resolution) training with PaddleOCR.

This script creates LMDB datasets from preprocessed_cropped data for training
SR models (Telescope/Gestalt). Each LMDB entry contains:
- label-%09d: plate text (UTF-8)
- image_hr-%09d: high-resolution image bytes (PNG)
- image_lr-%09d: low-resolution image bytes (PNG)
"""

import argparse
import lmdb
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from lib.data.track_loader import TrackLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_label(text):
    """Filter label to keep only alphanumeric (matching LMDBDataSetSR.str_filt)."""
    import string
    allowed = string.digits + string.ascii_letters
    return ''.join(c for c in text if c in allowed)


def get_clean_images(images):
    """Get non-augmented images sorted by name."""
    return sorted([img for img in set(images) if "_aug_" not in img.name], key=lambda x: x.name)


def collect_pairs(images_hr, images_lr, label, is_lr_as_hr=False):
    """Collect HR/LR pairs from image lists."""
    pairs = []
    
    # Match by base name (hr-001.png <-> lr-001.png)
    lr_dict = {img.stem.replace('lr-', ''): img for img in images_lr}
    
    for hr_img in images_hr:
        if is_lr_as_hr:
            base_name = hr_img.stem.replace('lr-', '')
        else:
            base_name = hr_img.stem.replace('hr-', '').split('_aug_')[0]
            
        lr_img = lr_dict.get(base_name)
        if lr_img:
            pairs.append({
                'hr_path': str(hr_img.resolve()),
                'lr_path': str(lr_img.resolve()),
                'label': label
            })
    
    return pairs


def build_lmdb(samples, output_dir, map_size=1099511627776):
    """Build LMDB database from samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = lmdb.open(str(output_dir), map_size=map_size)
    
    with env.begin(write=True) as txn:
        txn.put(b'num-samples', str(len(samples)).encode('utf-8'))
        
        for idx, sample in enumerate(tqdm(samples, desc=f"Building {output_dir.name}"), start=1):
            with open(sample['hr_path'], 'rb') as f:
                hr_bytes = f.read()
            with open(sample['lr_path'], 'rb') as f:
                lr_bytes = f.read()
            
            txn.put(f'label-{idx:09d}'.encode(), sample['label'].encode('utf-8'))
            txn.put(f'image_hr-{idx:09d}'.encode(), hr_bytes)
            txn.put(f'image_lr-{idx:09d}'.encode(), lr_bytes)
    
    env.close()
    logger.info(f"✓ Built LMDB: {output_dir} ({len(samples)} samples)")


def build_lmdb_sr_dataset(test_tracks_per_country=500, val_size=500):
    """Build LMDB datasets for SR training following multiframe split strategy."""
    
    raw_data_dir = PROJECT_ROOT / "data/preprocessed_cropped"
    output_dir = PROJECT_ROOT / "data/lmdb_sr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = TrackLoader(train_dir=raw_data_dir, logger=logger)
    all_tracks = loader.load_all_tracks()
    
    random.seed(42)
    
    scenario_a = [t for t in all_tracks if t.scenario == 'A']
    scenario_b = [t for t in all_tracks if t.scenario == 'B']
    logger.info(f"Tracks: {len(all_tracks)} (A: {len(scenario_a)}, B: {len(scenario_b)})")
    
    # === Test: 500 tracks/country from Scenario B, LR only ===
    tracks_by_country = defaultdict(list)
    for track in scenario_b:
        tracks_by_country[track.plate_layout.lower()].append(track)
    
    used_track_dirs = set()
    test_tracks = []
    for country, country_tracks in tracks_by_country.items():
        shuffled = country_tracks.copy()
        random.shuffle(shuffled)
        selected = shuffled[:min(test_tracks_per_country, len(shuffled))]
        test_tracks.extend(selected)
        for t in selected:
            used_track_dirs.add(str(t.track_dir))
        logger.info(f"Test ({country}): {len(selected)} tracks")
    
    # === Remaining tracks for train/val ===
    remaining_b = [t for t in scenario_b if str(t.track_dir) not in used_track_dirs]
    remaining = scenario_a + remaining_b
    random.shuffle(remaining)
    
    # Val: collect tracks until val_size pairs reached
    val_track_dirs = set()
    val_pair_count = 0
    for t in remaining:
        if val_pair_count >= val_size:
            break
        n = len(get_clean_images(t.lr_images))
        if n > 0:
            val_track_dirs.add(str(t.track_dir))
            val_pair_count += n
    
    val_tracks = [t for t in remaining if str(t.track_dir) in val_track_dirs]
    train_tracks = [t for t in remaining if str(t.track_dir) not in val_track_dirs]
    
    # === Collect samples ===
    train_samples = []
    for track in train_tracks:
        label = filter_label(track.plate_text)
        if not label:
            continue
        
        # HR clean
        train_samples.extend(collect_pairs(get_clean_images(track.hr_images), 
                                          get_clean_images(track.lr_images), label))
        
        # LR clean (paired with itself as "HR")
        train_samples.extend(collect_pairs(get_clean_images(track.lr_images),
                                          get_clean_images(track.lr_images), label, is_lr_as_hr=True))
    
    val_samples = []
    for track in val_tracks:
        label = filter_label(track.plate_text)
        if not label:
            continue
        val_samples.extend(collect_pairs(get_clean_images(track.lr_images),
                                        get_clean_images(track.lr_images), label, is_lr_as_hr=True))
    
    test_samples = []
    for track in test_tracks:
        label = filter_label(track.plate_text)
        if not label:
            continue
        test_samples.extend(collect_pairs(get_clean_images(track.lr_images),
                                         get_clean_images(track.lr_images), label, is_lr_as_hr=True))
    
    # === Build LMDB ===
    build_lmdb(train_samples, output_dir / 'train_lmdb')
    build_lmdb(val_samples, output_dir / 'val_lmdb')
    build_lmdb(test_samples, output_dir / 'test_lmdb')
    
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logger.info(f"Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build LMDB dataset for SR training')
    parser.add_argument('--test_tracks_per_country', type=int, default=500)
    parser.add_argument('--val_size', type=int, default=500)
    args = parser.parse_args()
    
    build_lmdb_sr_dataset(
        test_tracks_per_country=args.test_tracks_per_country,
        val_size=args.val_size
    )
