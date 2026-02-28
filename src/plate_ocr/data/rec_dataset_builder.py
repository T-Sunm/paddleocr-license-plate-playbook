"""
Recognition dataset builder for PaddleOCR format
All images are pre-cropped plates from preprocessed_cropped directory

Split logic:
- Test: LR images from 500 Scenario B tracks per country
- Train/Val: Remaining A+B samples split 9:1
"""
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from .sample import Sample
from .track_loader import TrackLoader, TrackInfo
from ..utils.file_utils import ensure_dir


class RecDatasetBuilder:
    def __init__(
        self, 
        output_dir: Path,
        train_ratio: float = 0.98,
        test_samples_per_country: int = 500,
        random_seed: int = 42,
        logger: logging.Logger = None
    ):
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.test_samples_per_country = test_samples_per_country
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)
        
        ensure_dir(self.output_dir)
        random.seed(self.random_seed)

    def _tracks_to_clean_samples(
        self, 
        tracks: List[TrackInfo], 
        track_loader: TrackLoader,
        resolution: str = None
    ) -> List[Sample]:
        """Convert tracks to samples without bbox (for recognition)"""
        if resolution:
            samples = track_loader.tracks_to_samples(tracks, resolution=resolution)
        else:
            samples_lr = track_loader.tracks_to_samples(tracks, resolution='lr')
            samples_hr = track_loader.tracks_to_samples(tracks, resolution='hr')
            samples = samples_lr + samples_hr
        
        return [
            Sample(
                image_path=s.image_path,
                text=s.text,
                country=s.country,
                scenario=s.scenario,
                resolution=s.resolution,
                points=None
            ) for s in samples
        ]

    def sample_test_set_from_scenario_b(
        self, 
        scenario_b_tracks: List[TrackInfo],
        track_loader: TrackLoader
    ) -> Tuple[List[Sample], Set[str]]:
        """Sample test set: 500 tracks per country, all original LR images per track"""
        tracks_by_country = defaultdict(list)
        for track in scenario_b_tracks:
            tracks_by_country[track.plate_layout.lower()].append(track)
        
        test_samples = []
        used_track_dirs = set()
        
        for country, country_tracks in tracks_by_country.items():
            shuffled = country_tracks.copy()
            random.shuffle(shuffled)
            
            num_tracks = min(self.test_samples_per_country, len(shuffled))
            selected_tracks = shuffled[:num_tracks]
            
            country_samples = []
            for track in selected_tracks:
                samples = track_loader.tracks_to_samples([track], resolution='lr')
                original = [s for s in samples if '_aug_' not in s.image_path.name]
                if original:
                    country_samples.extend(original)
                    used_track_dirs.add(str(track.track_dir))
            
            test_samples.extend(country_samples)
            self.logger.info(f"Test ({country}): {len(country_samples)} samples from {num_tracks} tracks")
        
        return test_samples, used_track_dirs

    def combine_and_split(
        self, 
        scenario_a_tracks: List[TrackInfo],
        scenario_b_tracks: List[TrackInfo],
        used_track_dirs: Set[str],
        track_loader: TrackLoader
    ) -> Tuple[List[Sample], List[Sample]]:
        """Combine A + B remaining. Val = 2% of LR only, Train = all HR + 98% LR"""
        # Scenario A: all samples
        samples_a = self._tracks_to_clean_samples(scenario_a_tracks, track_loader)
        
        # Scenario B: exclude test tracks
        remaining_b = [t for t in scenario_b_tracks if str(t.track_dir) not in used_track_dirs]
        samples_b = self._tracks_to_clean_samples(remaining_b, track_loader)
        
        all_samples = samples_a + samples_b
        
        # Split by resolution
        lr_samples = [s for s in all_samples if s.resolution == 'lr']
        hr_samples = [s for s in all_samples if s.resolution == 'hr']
        
        random.shuffle(lr_samples)
        val_size = 500
        
        val_samples = lr_samples[:val_size]
        train_samples = lr_samples[val_size:] + hr_samples
        random.shuffle(train_samples)
        
        self.logger.info(f"Total: {len(all_samples)} (LR: {len(lr_samples)}, HR: {len(hr_samples)})")
        
        return train_samples, val_samples

    def create_label_file(self, samples: List[Sample], output_file: Path) -> int:
        """Create PaddleOCR recognition label file (path\\ttext)"""
        ensure_dir(output_file.parent)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(f"{sample.image_path.resolve()}\t{sample.text}\n")
        
        self.logger.info(f"Created: {output_file.name} ({len(samples)} samples)")
        return len(samples)

    def build_recognition_dataset(
        self, 
        all_tracks: List[TrackInfo],
        track_loader: TrackLoader,
        train_label_name: str = "train_rec.txt",
        val_label_name: str = "val_rec.txt",
        test_label_name: str = "test_rec.txt"
    ) -> Dict:
        """Build complete recognition dataset from pre-cropped images"""
        scenario_a = [t for t in all_tracks if t.scenario == 'A']
        scenario_b = [t for t in all_tracks if t.scenario == 'B']
        
        self.logger.info(f"Tracks: {len(all_tracks)} (A: {len(scenario_a)}, B: {len(scenario_b)})")
        
        # Sample test set from Scenario B
        test_samples, used_tracks = self.sample_test_set_from_scenario_b(scenario_b, track_loader)
        
        # Combine and split train/val
        train_samples, val_samples = self.combine_and_split(
            scenario_a, scenario_b, used_tracks, track_loader
        )
        
        # Create label files
        train_count = self.create_label_file(train_samples, self.output_dir / train_label_name)
        val_count = self.create_label_file(val_samples, self.output_dir / val_label_name)
        test_count = self.create_label_file(test_samples, self.output_dir / test_label_name)
        
        self.logger.info("=" * 50)
        self.logger.info(f"Train: {train_count} | Val: {val_count} | Test: {test_count}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 50)
        
        return {
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "output_dir": str(self.output_dir)
        }
