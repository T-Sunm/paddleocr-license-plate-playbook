"""
Track-based data loader for plate recognition dataset
Loads data from Scenario-A and Scenario-B with track structure
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .sample import Sample


logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Information about a track"""
    track_dir: Path
    scenario: str  # 'A' or 'B'
    plate_layout: str  # 'Brazilian' or 'Mercosur'
    plate_text: str
    hr_images: List[Path]
    lr_images: List[Path]
    corners: Dict[str, Dict[str, List[int]]] = None
    

class TrackLoader:
    """
    Load tracks from Scenario-A and Scenario-B
    
    Directory structure:
    train/
        Scenario-A/
            Brazilian/
                track_XXXXX/
                    annotations.json
                    hr-001.png, hr-002.png, ...
                    lr-001.png, lr-002.png, ...
            Mercosur/
                track_XXXXX/
                    ...
        Scenario-B/
            Brazilian/
                track_XXXXX/
                    annotations.json
                    hr-001.jpg, hr-002.jpg, ...
                    lr-001.jpg, lr-002.jpg, ...
            Mercosur/
                track_XXXXX/
                    ...
    """
    
    def __init__(self, train_dir: Path, logger: logging.Logger = None):
        """
        Initialize track loader
        
        Args:
            train_dir: Path to train directory
            logger: Logger instance
        """
        self.train_dir = Path(train_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.train_dir.exists():
            raise ValueError(f"Train directory does not exist: {self.train_dir}")
    
    def load_track(self, track_dir: Path) -> Optional[TrackInfo]:
        """
        Load a single track
        
        Args:
            track_dir: Path to track directory
            
        Returns:
            TrackInfo object or None if failed
        """
        try:
            # Read annotations
            annotations_file = track_dir / "annotations.json"
            if not annotations_file.exists():
                self.logger.warning(f"No annotations.json in {track_dir}")
                return None
            
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            plate_text = annotations.get('plate_text', '')
            plate_layout = annotations.get('plate_layout', '')
            corners = annotations.get('corners', {})
            
            if 'Scenario-A' in str(track_dir):
                scenario = 'A'
            elif 'Scenario-B' in str(track_dir):
                scenario = 'B'
            else:
                self.logger.warning(f"Unknown scenario for {track_dir}")
                return None
            
            # Find HR and LR images (including augmented)
            img_exts = ['.png', '.jpg', '.jpeg']
            hr_images = []
            lr_images = []
            for ext in img_exts:
                hr_images.extend(track_dir.glob(f'hr-*{ext}'))
                hr_images.extend(track_dir.glob(f'hr-*_aug_*{ext}'))
                lr_images.extend(track_dir.glob(f'lr-*{ext}'))
                lr_images.extend(track_dir.glob(f'lr-*_aug_*{ext}'))
                
            hr_images = sorted(list(set(hr_images)))
            lr_images = sorted(list(set(lr_images)))
            
            if not hr_images and not lr_images:
                self.logger.warning(f"No images found in {track_dir}")
                return None
            
            return TrackInfo(
                track_dir=track_dir,
                scenario=scenario,
                plate_layout=plate_layout,
                plate_text=plate_text,
                hr_images=hr_images,
                lr_images=lr_images,
                corners=corners
            )
        except Exception as e:
            self.logger.error(f"Error loading track {track_dir}: {e}")
            return None
    
    def load_all_tracks(
        self, 
        scenario: Optional[str] = None,
        plate_layout: Optional[str] = None
    ) -> List[TrackInfo]:
        """
        Load all tracks from train directory
        
        Args:
            scenario: Filter by scenario ('A' or 'B'), None for all
            plate_layout: Filter by plate layout ('Brazilian' or 'Mercosur'), None for all
            
        Returns:
            List of TrackInfo objects
        """
        tracks = []
        
        # Determine which scenarios to load
        scenarios = []
        if scenario is None:
            scenarios = ['Scenario-A', 'Scenario-B']
        elif scenario.upper() == 'A':
            scenarios = ['Scenario-A']
        elif scenario.upper() == 'B':
            scenarios = ['Scenario-B']
        
        # Determine which plate layouts to load
        layouts = []
        if plate_layout is None:
            layouts = ['Brazilian', 'Mercosur']
        else:
            layouts = [plate_layout]
        
        # Load tracks
        for scenario_name in scenarios:
            scenario_dir = self.train_dir / scenario_name
            if not scenario_dir.exists():
                self.logger.warning(f"Scenario directory not found: {scenario_dir}")
                continue
            
            for layout in layouts:
                layout_dir = scenario_dir / layout
                if not layout_dir.exists():
                    self.logger.warning(f"Layout directory not found: {layout_dir}")
                    continue
                
                # Load all tracks in this layout
                for track_dir in sorted(layout_dir.iterdir()):
                    if not track_dir.is_dir():
                        continue
                    
                    track_info = self.load_track(track_dir)
                    if track_info:
                        tracks.append(track_info)
        
        self.logger.info(f"Loaded {len(tracks)} tracks")
        return tracks
    
    def tracks_to_samples(
        self, 
        tracks: List[TrackInfo],
        resolution: str = 'lr'
    ) -> List[Sample]:
        """
        Convert tracks to samples
        
        Args:
            tracks: List of TrackInfo objects
            resolution: 'hr' or 'lr'
            
        Returns:
            List of Sample objects
        """
        samples = []
        
        for track in tracks:
            # Select images based on resolution
            if resolution == 'lr':
                images = track.lr_images
            elif resolution == 'hr':
                images = track.hr_images
            else:
                raise ValueError(f"Invalid resolution: {resolution}. Must be 'hr' or 'lr'")
            
            # Create samples
            for img_path in images:
                # Extract corners for this image
                points = None
                if track.corners:
                    img_name = img_path.name
                    if img_name in track.corners:
                        img_corners = track.corners[img_name]
                        # Order: top-left, top-right, bottom-right, bottom-left
                        # PaddleOCR expects [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        # The json provides these keys
                        try:
                            tl = img_corners['top-left']
                            tr = img_corners['top-right']
                            br = img_corners['bottom-right']
                            bl = img_corners['bottom-left']
                            points = [tl, tr, br, bl]
                        except KeyError:
                            pass

                sample = Sample(
                    image_path=img_path,
                    text=track.plate_text,
                    country=track.plate_layout.lower(),  # Brazilian or Mercosur
                    scenario=track.scenario,
                    resolution=resolution,
                    points=points
                )
                samples.append(sample)
        
        self.logger.info(f"Created {len(samples)} samples from {len(tracks)} tracks")
        return samples
