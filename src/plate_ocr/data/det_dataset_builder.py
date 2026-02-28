"""
Dataset builder for PaddleOCR format
Converts raw samples to PaddleOCR training format with proper train/val/test split
"""
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import json

from .sample import Sample
from .track_loader import TrackLoader, TrackInfo
from ..utils.file_utils import copy_file_safe, get_unique_filename, ensure_dir


class DetDatasetBuilder:
    """
    Build PaddleOCR DET (Detection) format dataset from samples
    
    Split logic:
    - Train/Val: From Scenario A (90/10 split by default)
    - Test: 1000 images from Scenario B (LR only)
    """
    
    def __init__(
        self, 
        output_dir: Path,
        train_ratio: float = 0.9,
        test_samples_per_country: int = 500,
        random_seed: int = 42,
        logger: logging.Logger = None
    ):
        """
        Initialize dataset builder
        
        Args:
            output_dir: Output directory for the dataset
            train_ratio: Ratio of training samples from Scenario A
            test_samples_per_country: Number of test samples per country
            random_seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.train_ratio = train_ratio
        self.test_samples_per_country = test_samples_per_country
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)
        
        # Create directories
        ensure_dir(self.output_dir)
        
        # Set random seed
        random.seed(self.random_seed)
    
    def filter_scenario_a_tracks(self, tracks: List[TrackInfo]) -> List[TrackInfo]:
        """
        Filter tracks from Scenario A
        
        Args:
            tracks: List of all tracks
            
        Returns:
            List of Scenario A tracks
        """
        scenario_a = [t for t in tracks if t.scenario == 'A']
        self.logger.info(f"Scenario A tracks: {len(scenario_a)}")
        return scenario_a
    
    def filter_scenario_b_lr_tracks(self, tracks: List[TrackInfo]) -> Dict[str, List[TrackInfo]]:
        """
        Filter tracks from Scenario B, grouped by country
        
        Args:
            tracks: List of all tracks
            
        Returns:
            Dictionary mapping country to list of Scenario B tracks
        """
        scenario_b_tracks = defaultdict(list)
        
        for track in tracks:
            if track.scenario == 'B':
                scenario_b_tracks[track.plate_layout.lower()].append(track)
        
        # Log statistics
        for country, country_tracks in scenario_b_tracks.items():
            self.logger.info(f"Scenario B tracks ({country}): {len(country_tracks)}")
        
        return dict(scenario_b_tracks)
    
    def split_train_val_tracks(
        self, 
        tracks: List[TrackInfo]
    ) -> Tuple[List[TrackInfo], List[TrackInfo]]:
        """
        Split Scenario A tracks into train and validation sets
        
        Args:
            tracks: List of Scenario A tracks
            
        Returns:
            Tuple of (train_tracks, val_tracks)
        """
        # Shuffle with fixed seed
        shuffled = tracks.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * self.train_ratio)
        train_tracks = shuffled[:split_idx]
        val_tracks = shuffled[split_idx:]
        
        self.logger.info(
            f"Split - Train: {len(train_tracks)} tracks | Val: {len(val_tracks)} tracks"
        )
        
        return train_tracks, val_tracks
    
    def sample_test_set_tracks(
        self, 
        scenario_b_tracks: Dict[str, List[TrackInfo]],
        track_loader: TrackLoader
    ) -> List[Sample]:
        """
        Sample test set from Scenario B tracks (LR only)
        
        Args:
            scenario_b_tracks: Dictionary of country -> Scenario B tracks
            track_loader: TrackLoader instance to convert tracks to samples
            
        Returns:
            List of test samples (LR)
        """
        test_samples = []
        
        for country, country_tracks in scenario_b_tracks.items():
            # Shuffle country tracks
            shuffled = country_tracks.copy()
            random.shuffle(shuffled)
            
            # We need to collect enough SAMPLES, but we select by TRACKS
            # This is tricky because tracks have varying number of images.
            # Simple approach: shuffle tracks, take all LR images until we hit limit or run out.
            
            country_test_samples = []
            for track in shuffled:
                track_samples = track_loader.tracks_to_samples([track], resolution='lr')
                country_test_samples.extend(track_samples)
                
                if len(country_test_samples) >= self.test_samples_per_country:
                    # Truncate to exact number
                    country_test_samples = country_test_samples[:self.test_samples_per_country]
                    break
            
            test_samples.extend(country_test_samples)
            
            self.logger.info(
                f"Test set ({country}): {len(country_test_samples)} samples from {len(shuffled)} tracks"
            )
        
        return test_samples
    
    def create_label_file(
        self, 
        samples: List[Sample], 
        output_file: Path,
        prefix: str
    ) -> int:
        """
        Create PaddleOCR label file
        
        Format: 
            Plain Text: image_path\tplate_text
            Detailed:   image_path\t[{"transcription": "plate_text", "points": [[x1, y1], ...], "confidence": 1.0}]
        
        Args:
            samples: List of samples
            output_file: Output label file path
            prefix: Unused (kept for compatibility)
            
        Returns:
            Number of samples processed
        """
        ensure_dir(output_file.parent)
        
        processed_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Use absolute path
                abs_path = sample.image_path.resolve()
                
                # Write label in PaddleOCR format
                if sample.points:
                    # Detailed format with points
                    label_data = [{
                        "transcription": sample.text,
                        "points": sample.points
                    }]
                    label_str = json.dumps(label_data)
                    f.write(f"{abs_path}\t{label_str}\n")
                else:
                    # Fallback to simple format
                    f.write(f"{abs_path}\t{sample.text}\n")
                
                processed_count += 1
        
        self.logger.info(
            f"Created label file: {output_file.name} ({processed_count} samples)"
        )
        return processed_count
    
    def build_dataset(
        self, 
        all_tracks: List[TrackInfo],
        track_loader: TrackLoader,
        train_label_name: str = "train_label.txt",
        val_label_name: str = "val_label.txt",
        test_label_name: str = "test_label.txt"
    ) -> Dict:
        """
        Build complete PaddleOCR dataset with proper splits
        
        Logic:
        1. Filter Scenario A tracks -> split into train/val
        2. Filter Scenario B tracks -> sample test set (samples)
        3. Convert train/val tracks to samples
        4. Create label files for each split
        
        Args:
            all_tracks: List of all tracks
            track_loader: TrackLoader instance
            train_label_name: Training label file name
            val_label_name: Validation label file name
            test_label_name: Test label file name
            
        Returns:
            Dictionary with dataset statistics
        """
        self.logger.info("Starting dataset building...")
        self.logger.info(f"Total tracks: {len(all_tracks)}")
        
        # Step 1: Filter Scenario A and split train/val by TRACK
        scenario_a_tracks = self.filter_scenario_a_tracks(all_tracks)
        train_tracks, val_tracks = self.split_train_val_tracks(scenario_a_tracks)
        
        # Convert to samples (HR + LR for Train/Val)
        self.logger.info("Converting Train tracks to samples (HR + LR)...")
        train_samples_lr = track_loader.tracks_to_samples(train_tracks, resolution='lr')
        train_samples_hr = track_loader.tracks_to_samples(train_tracks, resolution='hr')
        train_samples = train_samples_lr + train_samples_hr
        
        self.logger.info("Converting Val tracks to samples (HR + LR)...")
        val_samples_lr = track_loader.tracks_to_samples(val_tracks, resolution='lr')
        val_samples_hr = track_loader.tracks_to_samples(val_tracks, resolution='hr')
        val_samples = val_samples_lr + val_samples_hr
        
        # Step 2: Filter Scenario B and sample test set
        scenario_b_tracks = self.filter_scenario_b_lr_tracks(all_tracks)
        test_samples = self.sample_test_set_tracks(scenario_b_tracks, track_loader)
        
        # Step 3: Create label files
        train_path = self.output_dir / train_label_name
        val_path = self.output_dir / val_label_name
        test_path = self.output_dir / test_label_name
        
        train_count = self.create_label_file(train_samples, train_path, "train")
        val_count = self.create_label_file(val_samples, val_path, "val")
        test_count = self.create_label_file(test_samples, test_path, "test")
        
        # Compile statistics
        stats = {
            "total_tracks": len(all_tracks),
            "scenario_a_tracks": len(scenario_a_tracks),
            "scenario_b_tracks": sum(len(v) for v in scenario_b_tracks.values()),
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "test_samples_per_country": {
                country: len([s for s in test_samples if s.country == country])
                for country in scenario_b_tracks.keys()
            },
            "output_dir": str(self.output_dir),
            "train_label": str(train_path),
            "val_label": str(val_path),
            "test_label": str(test_path),
        }
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Dataset Building Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total tracks: {stats['total_tracks']}")
        self.logger.info(f"Scenario A tracks: {stats['scenario_a_tracks']}")
        self.logger.info(f"Scenario B tracks: {stats['scenario_b_tracks']}")
        self.logger.info("-" * 60)
        self.logger.info(f"Train samples: {stats['train_samples']}")
        self.logger.info(f"Val samples: {stats['val_samples']}")
        self.logger.info(f"Test samples: {stats['test_samples']}")
        self.logger.info("-" * 60)
        self.logger.info("Test samples per country:")
        for country, count in stats['test_samples_per_country'].items():
            self.logger.info(f"  {country}: {count}")
        self.logger.info("=" * 60)
        
        return stats
