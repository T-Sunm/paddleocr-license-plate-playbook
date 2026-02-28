"""
Script to build datasets for PaddleOCR (Detection & Recognition)
Unified script that reads from shared config file.

Usage:
    python scripts/build_dataset.py --mode det   # Build detection dataset
    python scripts/build_dataset.py --mode rec   # Build recognition dataset
    python scripts/build_dataset.py --mode all   # Build both
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from plate_ocr.paths import PROJECT_ROOT
from plate_ocr.data.track_loader import TrackLoader
from plate_ocr.data.det_dataset_builder import DetDatasetBuilder
from plate_ocr.data.rec_dataset_builder import RecDatasetBuilder


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config():
    config_path = PROJECT_ROOT / "shared/configs/dataset_builder.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default configuration if file missing
    return {
        'common': {
            'raw_data_dir': 'data/preprocessed_cropped',
            'train_ratio': 0.98,
            'test_samples_per_country': 500,
            'random_seed': 42
        },
        'detection': {
            'output_dir': 'data/processed_det',
            'labels': {'train': 'train_det.txt', 'val': 'val_det.txt', 'test': 'test_det.txt'}
        },
        'recognition': {
            'output_dir': 'data/processed_rec',
            'labels': {'train': 'train_rec.txt', 'val': 'val_rec.txt', 'test': 'test_rec.txt'}
        }
    }


def build_detection(config: dict, track_loader, all_tracks, logger):
    """Build detection dataset"""
    common = config['common']
    det_cfg = config['detection']
    
    output_dir = PROJECT_ROOT / det_cfg['output_dir']
    
    logger.info("=" * 60)
    logger.info("Building DETECTION dataset...")
    logger.info("=" * 60)
    
    builder = DetDatasetBuilder(
        output_dir=output_dir,
        train_ratio=common['train_ratio'],
        test_samples_per_country=common['test_samples_per_country'],
        random_seed=common['random_seed'],
        logger=logger
    )
    
    stats = builder.build_dataset(
        all_tracks=all_tracks,
        track_loader=track_loader,
        train_label_name=det_cfg['labels']['train'],
        val_label_name=det_cfg['labels']['val'],
        test_label_name=det_cfg['labels']['test']
    )
    
    logger.info(f"Detection dataset saved to: {stats['output_dir']}")
    return stats


def build_recognition(config: dict, track_loader, all_tracks, logger):
    """Build recognition dataset"""
    common = config['common']
    rec_cfg = config['recognition']
    
    output_dir = PROJECT_ROOT / rec_cfg['output_dir']
    
    logger.info("=" * 60)
    logger.info("Building RECOGNITION dataset...")
    logger.info("=" * 60)
    
    builder = RecDatasetBuilder(
        output_dir=output_dir,
        train_ratio=common['train_ratio'],
        test_samples_per_country=common['test_samples_per_country'],
        random_seed=common['random_seed'],
        logger=logger
    )
    
    stats = builder.build_recognition_dataset(
        all_tracks=all_tracks,
        track_loader=track_loader,
        train_label_name=rec_cfg['labels']['train'],
        val_label_name=rec_cfg['labels']['val'],
        test_label_name=rec_cfg['labels']['test']
    )
    
    logger.info(f"Recognition dataset saved to: {stats['output_dir']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build dataset for PaddleOCR")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["det", "rec", "all"], 
        default="all",
        help="Which dataset to build: det (detection), rec (recognition), or all"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    config = load_config()
    
    # Load tracks (shared for both)
    raw_data_dir = PROJECT_ROOT / config['common']['raw_data_dir']
    logger.info(f"Loading tracks from: {raw_data_dir}")
    
    track_loader = TrackLoader(train_dir=raw_data_dir, logger=logger)
    all_tracks = track_loader.load_all_tracks()
    logger.info(f"Total tracks loaded: {len(all_tracks)}")
    
    # Build datasets based on mode
    if args.mode in ["det", "all"]:
        build_detection(config, track_loader, all_tracks, logger)
    
    if args.mode in ["rec", "all"]:
        build_recognition(config, track_loader, all_tracks, logger)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset building completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
