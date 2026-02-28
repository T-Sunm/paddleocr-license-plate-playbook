#!/usr/bin/env python3
"""
Preprocess images for Scenario A/B:
1. Resize all images to TARGET_SIZE using Albumentations
2. Scenario A: Transform existing corners with resize
3. Scenario B: Detect bbox using Qwen3-VL
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

TARGET_SIZE = (336, 1008)  # (height, width)
QWEN_SCALE_FACTOR = 1000
IMAGE_EXTENSIONS = {".jpg", ".png"}
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

DETECTION_PROMPT = """
Task: Detect the precise license plate text region.
Step 1: Analyze the image and identify the exact pixels where the text starts (left) and ends (right).
Step 2: Identify the top and bottom edges of the characters.
Step 3: Reason about the boundaries - ensure no background is included, but no text is cut.
Step 4: Output the result.

Return JSON format:
{"bbox_2d": [x1, y1, x2, y2]}
"""

BBox = Tuple[int, int, int, int]
Corners = Dict[str, List[int]]

resize_transform = A.Compose([
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


class QwenDetector:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_id: str = MODEL_ID):
        if cls._instance is None:
            cls._instance = cls(model_id)
        return cls._instance
    
    def __init__(self, model_id: str = MODEL_ID):
        print(f"Loading Qwen3-VL: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,    # Flash Attention yêu cầu fp16 hoặc bf16
            device_map="cuda",
            attn_implementation="flash_attention_2"
        )

    def detect(self, image_path: Path) -> Optional[Corners]:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return self.detect_from_array(img)
    
    def detect_from_array(self, img: np.ndarray) -> Optional[Corners]:
        """Detect bbox from numpy array image"""
        raw_text = self._run_inference(img)
        bbox = self._parse_output(raw_text)
        
        if bbox:
            h, w = img.shape[:2]
            final_bbox = self._normalize_bbox(bbox, w, h)
            return self._format_corners(final_bbox)
        return None

    def _run_inference(self, img: np.ndarray) -> str:
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": DETECTION_PROMPT}
        ]}]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

        text = self.processor.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip().replace("```json", "").replace("```", "").strip()

    def _parse_output(self, text: str) -> Optional[List[int]]:
        try:
            data = json.loads(text)
            bbox = self._extract_bbox_recursive(data)
            if bbox and len(bbox) == 4:
                return bbox
        except json.JSONDecodeError:
            pass

        match = re.search(r'bbox_2d["\s:]+\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)', text)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]
        return None

    def _extract_bbox_recursive(self, data: Union[dict, list]) -> Optional[List[int]]:
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            if "bbox_2d" in data:
                return data["bbox_2d"]
            for value in data.values():
                res = self._extract_bbox_recursive(value)
                if res:
                    return res
        return None

    def _normalize_bbox(self, bbox: List[int], width: int, height: int) -> BBox:
        x1 = int(bbox[0] / QWEN_SCALE_FACTOR * width)
        y1 = int(bbox[1] / QWEN_SCALE_FACTOR * height)
        x2 = int(bbox[2] / QWEN_SCALE_FACTOR * width)
        y2 = int(bbox[3] / QWEN_SCALE_FACTOR * height)
        return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))

    def _format_corners(self, bbox: BBox) -> Corners:
        x1, y1, x2, y2 = bbox
        return {
            "top-left": [x1, y1], "top-right": [x2, y1],
            "bottom-right": [x2, y2], "bottom-left": [x1, y2]
        }


def get_images(track_dir: Path) -> List[Path]:
    return sorted([
        f for f in track_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        and (f.name.startswith("lr-") or f.name.startswith("hr-"))
    ])


def corners_to_bbox(corners: Corners, img_width: int, img_height: int) -> List[int]:
    """Convert 4-corner format to pascal_voc bbox [x_min, y_min, x_max, y_max]."""
    xs = [corners[k][0] for k in corners]
    ys = [corners[k][1] for k in corners]
    
    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(img_width - 1, max(xs))
    y_max = min(img_height - 1, max(ys))
    
    return [x_min, y_min, x_max, y_max]


def bbox_to_corners(bbox: Tuple[float, float, float, float]) -> Corners:
    """Convert pascal_voc bbox to 4-corner format."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return {
        "top-left": [x1, y1], "top-right": [x2, y1],
        "bottom-right": [x2, y2], "bottom-left": [x1, y2]
    }


def resize_with_bbox(img: np.ndarray, bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[Corners]]:
    """Resize image and transform bbox using Albumentations."""
    try:
        if bbox:
            result = resize_transform(image=img, bboxes=[bbox], labels=["plate"])
            new_corners = bbox_to_corners(result['bboxes'][0]) if result['bboxes'] else None
            return result['image'], new_corners
        else:
            result = resize_transform(image=img, bboxes=[], labels=[])
            return result['image'], None
    except Exception as e:
        tqdm.write(f"[WARN] Albumentations Error during bbox resize: {e} - falling back to no-bbox")
        result = resize_transform(image=img, bboxes=[], labels=[])
        return result['image'], None


def process_track_a(track_dir: Path, output_dir: Path, detector: Optional[QwenDetector], dry_run: bool) -> Tuple[bool, str]:
    ann_file = track_dir / "annotations.json"
    if not ann_file.exists():
        return False, "No annotations.json"

    with open(ann_file) as f:
        ann = json.load(f)

    output_ann = output_dir / "annotations.json"
    if output_ann.exists():
        return True, "Already processed"

    images = get_images(track_dir)
    if not images:
        return False, "No images"

    output_dir.mkdir(parents=True, exist_ok=True)
    new_corners = {}
    corners_data = ann.get("corners", {})

    for img_path in images:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            img_corners = corners_data.get(img_path.name)
            
            if img_corners:
                bbox = corners_to_bbox(img_corners, w, h)
                img_resized, transformed_corners = resize_with_bbox(img, bbox)
                if transformed_corners:
                    new_corners[img_path.name] = transformed_corners
            else:
                img_resized, _ = resize_with_bbox(img, None)
                if detector:
                    result = detector.detect_from_array(img_resized)
                    if result:
                        new_corners[img_path.name] = result
            
            if not dry_run:
                cv2.imwrite(str(output_dir / img_path.name), img_resized)
        except Exception as e:
            tqdm.write(f"[WARN] {track_dir.name}/{img_path.name}: {str(e)}")
            continue

    ann["corners"] = new_corners
    if not dry_run:
        with open(output_ann, 'w') as f:
            json.dump(ann, f, indent=2)

    return True, f"{len(images)} images"


def process_track_b(detector: QwenDetector, track_dir: Path, 
                    output_dir: Path, dry_run: bool) -> Tuple[bool, str]:
    ann_file = track_dir / "annotations.json"
    if not ann_file.exists():
        return False, "No annotations.json"

    with open(ann_file) as f:
        ann = json.load(f)

    output_ann = output_dir / "annotations.json"
    if output_ann.exists():
        with open(output_ann) as f:
            if json.load(f).get("corners"):
                return True, "Already processed"

    images = get_images(track_dir)
    if not images:
        return False, "No images"

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized, _ = resize_with_bbox(img)
        out_path = output_dir / img_path.name
        if not dry_run:
            cv2.imwrite(str(out_path), img_resized)
        saved_paths.append(out_path)

    if not saved_paths:
        return False, "No images saved"

    corners_map = {}
    for img_path in saved_paths:
        result = detector.detect(img_path)
        if result:
            corners_map[img_path.name] = result

    if not corners_map:
        return False, "Detection failed"

    ann["corners"] = corners_map
    if not dry_run:
        with open(output_ann, 'w') as f:
            json.dump(ann, f, indent=2)

    return True, f"{len(saved_paths)} images, {len(corners_map)} corners"


def process_scenario(scenario_dir: Path, output_dir: Path, detector: Optional[QwenDetector],
                     is_scenario_b: bool, layout: str, limit: Optional[int], dry_run: bool) -> Dict[str, int]:
    stats = {"success": 0, "skip": 0, "fail": 0}
    
    layouts = [scenario_dir / layout] if layout != "all" else [d for d in scenario_dir.iterdir() if d.is_dir()]

    for layout_dir in layouts:
        if not layout_dir.exists():
            continue

        tracks = sorted([d for d in layout_dir.iterdir() if d.is_dir() and d.name.startswith("track_")])
        if limit:
            tracks = tracks[:limit]

        print(f"{layout_dir.name}: {len(tracks)} tracks")

        for track in tqdm(tracks, desc=layout_dir.name):
            output_track = output_dir / layout_dir.name / track.name
            
            if is_scenario_b:
                success, msg = process_track_b(detector, track, output_track, dry_run)
            else:
                success, msg = process_track_a(track, output_track, detector, dry_run)
            
            if success:
                stats["skip" if "Already" in msg else "success"] += 1
            else:
                stats["fail"] += 1
                tqdm.write(f"[FAIL] {track.name}: {msg}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess Scenario A/B images")
    parser.add_argument("--data-dir", required=True, type=Path, help="Path to data/raw/train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--layout", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenario", choices=["A", "B", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Path not found: {args.data_dir}")
        return

    output_base = args.output_dir if args.output_dir.is_absolute() else args.data_dir.parent.parent / args.output_dir.name
    
    print(f"Input: {args.data_dir}")
    print(f"Output: {output_base}")
    print(f"Size: {TARGET_SIZE[1]}x{TARGET_SIZE[0]}")

    total = {"success": 0, "skip": 0, "fail": 0}
    detector = None

    if args.scenario in ["A", "all"]:
        scenario_a = args.data_dir / "Scenario-A"
        if scenario_a.exists():
            print("\n[Scenario-A] resize + transform corners (fallback Qwen if no bbox)")
            if detector is None:
                detector = QwenDetector.get_instance()
            stats = process_scenario(scenario_a, output_base / "Scenario-A", detector, False, args.layout, args.limit, args.dry_run)
            for k in total:
                total[k] += stats[k]

    if args.scenario in ["B", "all"]:
        scenario_b = args.data_dir / "Scenario-B"
        if scenario_b.exists():
            print("\n[Scenario-B] resize + Qwen3 detection")
            detector = QwenDetector.get_instance()
            stats = process_scenario(scenario_b, output_base / "Scenario-B", detector, True, args.layout, args.limit, args.dry_run)
            for k in total:
                total[k] += stats[k]

    print(f"\nResults: success={total['success']}, skip={total['skip']}, fail={total['fail']}")


if __name__ == "__main__":
    main()
