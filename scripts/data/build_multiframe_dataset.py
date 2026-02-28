import random
import logging
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import yaml

from plate_ocr.paths import PROJECT_ROOT
from plate_ocr.data.track_loader import TrackLoader
from plate_ocr.utils.file_utils import ensure_dir
from _dataset_utils import get_clean_images, get_aug_groups, write_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_multiframe_dataset(seq_len=5, test_tracks_per_country=500, val_size=500):
    raw_data_dir = PROJECT_ROOT / "data" / "preprocessed_cropped"
    output_dir = PROJECT_ROOT / "data" / "processed_rec_multiframe"
    ensure_dir(output_dir)

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

    # Val: collect tracks until val_size LR sequences reached
    val_track_dirs = set()
    val_seq_count = 0
    for t in remaining:
        if val_seq_count >= val_size:
            break
        n = len(get_clean_images(t.lr_images)) // seq_len
        if n > 0:
            val_track_dirs.add(str(t.track_dir))
            val_seq_count += n

    val_tracks = [t for t in remaining if str(t.track_dir) in val_track_dirs]
    train_tracks = [t for t in remaining if str(t.track_dir) not in val_track_dirs]

    # === Write train.txt: HR (clean + augmented) + LR (clean) ===
    train_count = 0
    with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
        for track in train_tracks:
            train_count += write_sequences(f, get_clean_images(track.hr_images), track.plate_text, seq_len)
            for aug_list in get_aug_groups(track.hr_images).values():
                train_count += write_sequences(f, aug_list, track.plate_text, seq_len)
            train_count += write_sequences(f, get_clean_images(track.lr_images), track.plate_text, seq_len)
    logger.info(f"Train: {train_count} sequences")

    # === Write val.txt: LR only (clean) ===
    val_count = 0
    with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
        for track in val_tracks:
            val_count += write_sequences(f, get_clean_images(track.lr_images), track.plate_text, seq_len)
    logger.info(f"Val: {val_count} sequences")

    # === Write test.txt: LR only (clean) ===
    test_count = 0
    with open(output_dir / "test.txt", 'w', encoding='utf-8') as f:
        for track in test_tracks:
            test_count += write_sequences(f, get_clean_images(track.lr_images), track.plate_text, seq_len)
    logger.info(f"Test: {test_count} sequences")

    logger.info("=" * 50)
    logger.info(f"Train: {train_count} | Val: {val_count} | Test: {test_count}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build multiframe dataset for recognition")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--test_tracks_per_country", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=500)
    args = parser.parse_args()
    build_multiframe_dataset(
        seq_len=args.seq_len,
        test_tracks_per_country=args.test_tracks_per_country,
        val_size=args.val_size
    )
