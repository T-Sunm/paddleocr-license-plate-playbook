"""
File utility functions for dataset building
"""
import shutil
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def copy_file_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def get_unique_filename(
    directory: Path, 
    prefix: str, 
    index: int, 
    extension: str
) -> str:
    """
    Generate a unique filename
    
    Args:
        directory: Target directory
        prefix: Filename prefix
        index: Index number
        extension: File extension (with or without dot)
        
    Returns:
        Unique filename string
    """
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Generate base filename
    filename = f"{prefix}_{index:06d}{extension}"
    
    # Check if file exists and generate new name if needed
    counter = 0
    while (directory / filename).exists():
        counter += 1
        filename = f"{prefix}_{index:06d}_{counter}{extension}"
    
    return filename


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_files(directory: Path, pattern: str = "*") -> int:
    """
    Count files in directory matching pattern
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        
    Returns:
        Number of matching files
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))
