"""
Script to crop license plates from images using bounding box annotations.
Saves cropped images to preprocessed_cropped folder with same directory structure.
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm


def get_bounding_rect(corners: dict, expand: int = 10) -> tuple:
    """Convert 4-corner annotation to bounding rectangle with optional expansion."""
    pts = np.array([
        corners["top-left"],
        corners["top-right"],
        corners["bottom-right"],
        corners["bottom-left"]
    ], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    return x - expand, y - expand, x + w + expand, y + h + expand


def crop_and_save(img_path: Path, corners: dict, output_path: Path) -> bool:
    """Crop image using corners and save to output path."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    x1, y1, x2, y2 = get_bounding_rect(corners)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    cropped = img[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped)
    return True


def process_track(track_dir: Path, output_base: Path, input_base: Path) -> dict:
    """Process a single track directory."""
    ann_file = track_dir / "annotations.json"
    if not ann_file.exists():
        return {"track": str(track_dir), "status": "no_annotation"}
    
    with open(ann_file) as f:
        ann = json.load(f)
    
    rel_path = track_dir.relative_to(input_base)
    output_dir = output_base / rel_path
    
    corners_data = ann.get("corners", {})
    processed = 0
    failed = 0
    skipped = 0
    
    for img_name, corners in corners_data.items():
        img_path = track_dir / img_name
        if not img_path.exists():
            failed += 1
            continue
        
        output_path = output_dir / img_name
        
        # Skip if already cropped
        if output_path.exists():
            processed += 1
            skipped += 1
            continue
        
        if crop_and_save(img_path, corners, output_path):
            processed += 1
        else:
            failed += 1
    
    # Copy annotation with plate_text info
    output_ann = {
        "plate_text": ann.get("plate_text", ""),
        "plate_layout": ann.get("plate_layout", "")
    }
    output_ann_path = output_dir / "annotations.json"
    output_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ann_path, "w") as f:
        json.dump(output_ann, f, indent=2)
    
    return {"track": str(rel_path), "processed": processed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Crop plates from images using bbox annotations")
    parser.add_argument("--input", type=str, 
                        default="/home/temp-user/workspace/plate-recognition-low-resolution/data/preprocessed",
                        help="Input preprocessed directory")
    parser.add_argument("--output", type=str,
                        default="/home/temp-user/workspace/plate-recognition-low-resolution/data/preprocessed_cropped",
                        help="Output directory for cropped images")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    input_base = Path(args.input)
    output_base = Path(args.output)
    
    # Find all track directories
    tracks = list(input_base.glob("**/track_*"))
    print(f"Found {len(tracks)} tracks to process")
    
    results = {"processed": 0, "failed": 0, "no_annotation": 0}
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_track, t, output_base, input_base): t for t in tracks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping"):
            result = future.result()
            if result.get("status") == "no_annotation":
                results["no_annotation"] += 1
            else:
                results["processed"] += result.get("processed", 0)
                results["failed"] += result.get("failed", 0)
    
    print(f"\nDone!")
    print(f"Cropped images: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"Tracks without annotation: {results['no_annotation']}")
    print(f"Output saved to: {output_base}")


if __name__ == "__main__":
    main()
