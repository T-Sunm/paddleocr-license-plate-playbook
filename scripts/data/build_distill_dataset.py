import random
import logging
import argparse
import sys
from pathlib import Path
import yaml

from plate_ocr.paths import PROJECT_ROOT
from plate_ocr.data.track_loader import TrackLoader
from plate_ocr.utils.file_utils import ensure_dir
from _dataset_utils import get_clean_images, write_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    config_path = PROJECT_ROOT / "shared/configs/dataset_builder.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        'common': {
            'raw_data_dir': 'data/preprocessed_cropped'
        }
    }


def build_distill_dataset(mode="teacher_hr", seq_len=5):
    config = load_config()
    raw_data_dir = PROJECT_ROOT / config['common']['raw_data_dir']
    output_dir = PROJECT_ROOT / "data/processed_rec_distill"
    ensure_dir(output_dir)
    
    loader = TrackLoader(train_dir=raw_data_dir, logger=logger)
    all_tracks = loader.load_all_tracks()
    
    random.seed(42)
    random.shuffle(all_tracks)
    
    split_idx = int(len(all_tracks) * 0.98)
    splits = {"train": all_tracks[:split_idx], "val": all_tracks[split_idx:]}
    
    for split_name, tracks in splits.items():
        count = 0
        label_file = output_dir / f"{split_name}_{mode}.txt"
        
        with open(label_file, 'w', encoding='utf-8') as f:
            for track in tracks:
                if mode == "teacher_hr":
                    hr_clean = get_clean_images(track.hr_images)
                    
                    if split_name == "train":
                        # Train: thêm augmented groups
                        aug_groups = {}
                        for img in set(track.hr_images):
                            if "_aug_" in img.name:
                                suffix = img.name.split("_aug_")[1].split(".")[0]
                                aug_groups.setdefault(suffix, []).append(img)
                        all_lists = [hr_clean] + [sorted(g, key=lambda x: x.name) for g in aug_groups.values()]
                    else:
                        all_lists = [hr_clean]
                    
                    for hr_list in all_lists:
                        count += write_sequences(f, hr_list, track.plate_text, seq_len)
                        
                elif mode == "student_pair":
                    hr_clean = get_clean_images(track.hr_images)
                    lr_clean = get_clean_images(track.lr_images)
                    
                    for i in range(0, min(len(hr_clean), len(lr_clean)) - seq_len + 1, seq_len):
                        hr_paths = ",".join([str(p.resolve()) for p in hr_clean[i:i+seq_len]])
                        lr_paths = ",".join([str(p.resolve()) for p in lr_clean[i:i+seq_len]])
                        f.write(f"{lr_paths}|{hr_paths}\t{track.plate_text}\n")
                        count += 1
                        
        logger.info(f"Created {split_name}_{mode}.txt ({count} sequences)")
        
        # Val LR-only for eval
        if mode == "student_pair" and split_name == "val":
            lr_file = output_dir / f"{split_name}_student_lr.txt"
            with open(lr_file, 'w', encoding='utf-8') as f:
                for track in tracks:
                    write_sequences(f, get_clean_images(track.lr_images), track.plate_text, seq_len)
            logger.info(f"Created {lr_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teacher_hr", "student_pair"], default="teacher_hr")
    parser.add_argument("--seq_len", type=int, default=5)
    args = parser.parse_args()
    build_distill_dataset(mode=args.mode, seq_len=args.seq_len)
