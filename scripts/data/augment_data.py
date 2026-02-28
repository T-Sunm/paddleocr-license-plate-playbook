import os
import sys
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/home/temp-user/workspace/plate-recognition-low-resolution/straug')

from straug.weather import Snow, Rain
from straug.camera import Brightness, LRCompression
from straug.geometry import Rotate, Perspective

# OLD imports (noise-based augmentation)
# from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
# from straug.camera import JpegCompression, Pixelate

# Configuration
INPUT_ROOT = Path("/home/temp-user/workspace/plate-recognition-low-resolution/data/preprocessed")
OUTPUT_ROOT = INPUT_ROOT
NUM_AUGMENTATIONS_PER_IMAGE = 1


class PlateAugmentor:
    """
    LRLPR Pipeline:
    - Geometry(50%): Rotate or Perspective
    - Brightness → LRCompression
    - Weather(70%): only if no Geometry
    """
    def __init__(self):
        self.brightness = Brightness()
        self.lr_comp = LRCompression()
        self.weather = [Snow(), Rain()]
        self.geometry = [(Rotate(), 0), (Perspective(), 1)]  # (aug, mag)

    def __call__(self, img):
        has_geometry = False
        
        # 1. Geometry (50% chance)
        if np.random.uniform(0, 1) < 0.5:
            geo_idx = np.random.randint(0, 2)
            geo_aug, geo_mag = self.geometry[geo_idx]
            img = geo_aug(img, mag=geo_mag, prob=1.0)
            has_geometry = True
        
        # 2. Brightness
        img = self.brightness(img, mag=0, prob=1.0)
        
        # 3. LRCompression
        img = self.lr_comp(img, mag=0, prob=1.0)
        
        # 4. Weather (70% chance) - SKIP if Geometry was applied
        if not has_geometry:
            weather_aug = self.weather[np.random.randint(0, 2)]
            img = weather_aug(img, mag=2, prob=0.7)
        
        return img


def is_track_augmented(track_dir: Path) -> bool:
    """Check if track already has augmented files"""
    return any(track_dir.glob('*_aug_*.png')) or any(track_dir.glob('*_aug_*.jpg'))


def should_augment_track(track_dir: Path) -> bool:
    """Only augment if plate_text contains K, L, M, N, or O"""
    annotation_file = track_dir / 'annotations.json'
    if not annotation_file.exists():
        return False
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        plate_text = data.get('plate_text', '')
        return any(c in plate_text.upper() for c in 'KLMNO')
    except:
        return False


def augment_dataset(limit_tracks=None):
    augmentor = PlateAugmentor()
    from collections import defaultdict
    
    # Group tracks by scenario/layout (e.g. Scenario-A/Mercosur)
    groups = defaultdict(list)
    for track in INPUT_ROOT.rglob('track_*'):
        if track.is_dir():
            # Get parent path relative to INPUT_ROOT (e.g. Scenario-A/Mercosur)
            group_key = track.parent.relative_to(INPUT_ROOT)
            groups[group_key].append(track)
    
    # Sort and limit tracks per group
    selected_tracks = []
    skipped_augmented = 0
    skipped_filter = 0
    for group_key in sorted(groups.keys()):
        tracks = sorted(groups[group_key])
        if limit_tracks:
            tracks = tracks[:limit_tracks]
        
        # Filter out already augmented tracks and tracks not matching KLMNO filter
        tracks_to_process = []
        group_aug = 0
        group_filter = 0
        for track in tracks:
            if is_track_augmented(track):
                group_aug += 1
            elif not should_augment_track(track):
                group_filter += 1
            else:
                tracks_to_process.append(track)
        
        skipped_augmented += group_aug
        skipped_filter += group_filter
        selected_tracks.extend(tracks_to_process)
        print(f"  {group_key}: {len(tracks_to_process)} tracks (skip: {group_aug} augmented, {group_filter} no KLMNO)")
    
    print(f"Total: {len(selected_tracks)} to augment (skip: {skipped_augmented} augmented, {skipped_filter} no KLMNO)")
    
    image_files = []
    for track in selected_tracks:
        # Support both .png (Scenario-A) and .jpg (Scenario-B)
        for file in list(track.glob('hr-*.png')) + list(track.glob('hr-*.jpg')):
            image_files.append(str(file))

    print(f"Found {len(image_files)} HR images. Starting augmentation...")

    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert('RGB')
            rel_path = os.path.relpath(img_path, INPUT_ROOT)
            output_dir = OUTPUT_ROOT / os.path.dirname(rel_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            original_name = os.path.basename(img_path)
            name, ext = os.path.splitext(original_name)

            for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                augmented = augmentor(img.copy())
                new_filename = f"{name}_aug_{i}{ext}"
                augmented.save(output_dir / new_filename)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(INPUT_ROOT), help='Input directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tracks to process')
    args = parser.parse_args()
    INPUT_ROOT = Path(args.input)
    OUTPUT_ROOT = INPUT_ROOT
    augment_dataset(limit_tracks=args.limit)
