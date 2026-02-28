"""
Data structures for dataset building
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Sample:
    """
    Represents a single data sample
    
    Attributes:
        image_path: Path to the image file
        text: Plate text/label
        country: Country code (e.g., 'vietnam', 'thailand')
        scenario: Scenario type ('A' or 'B')
        resolution: Resolution type ('hr' or 'lr')
    """
    image_path: Path
    text: str
    country: str
    scenario: str
    resolution: str
    points: Optional[list] = None
    
    def __post_init__(self):
        """Validate and convert types"""
        self.image_path = Path(self.image_path)
        self.scenario = self.scenario.upper()
        self.resolution = self.resolution.lower()
        
        # Validate scenario
        if self.scenario not in ['A', 'B']:
            raise ValueError(f"Invalid scenario: {self.scenario}. Must be 'A' or 'B'")
        
        # Validate resolution
        if self.resolution not in ['hr', 'lr']:
            raise ValueError(f"Invalid resolution: {self.resolution}. Must be 'hr' or 'lr'")
    
    def is_scenario_a(self) -> bool:
        """Check if sample is from Scenario A"""
        return self.scenario == 'A'
    
    def is_scenario_b(self) -> bool:
        """Check if sample is from Scenario B"""
        return self.scenario == 'B'
    
    def is_low_resolution(self) -> bool:
        """Check if sample is low resolution"""
        return self.resolution == 'lr'
    
    def is_high_resolution(self) -> bool:
        """Check if sample is high resolution"""
        return self.resolution == 'hr'
    
    def __repr__(self) -> str:
        return (f"Sample(country={self.country}, scenario={self.scenario}, "
                f"resolution={self.resolution}, text={self.text})")
