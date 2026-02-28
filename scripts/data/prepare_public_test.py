import cv2
import torch
import numpy as np
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# Fixed Paths & Config
INPUT_DIR = Path("data/public_test")
OUTPUT_DIR = Path("data/public_test_cropped")
TARGET_SIZE = (336, 1008)
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DETECTION_PROMPT = "Task: Detect the precise license plate text region.\nReturn JSON format: {\"bbox_2d\": [x1, y1, x2, y2]}"

class PublicTestProcessor:
    def __init__(self):
        print(f"Loading {MODEL_ID} on GPU...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="cuda", 
            attn_implementation="flash_attention_2"
        )

    def detect_bbox(self, img_array) -> list:
        image_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": DETECTION_PROMPT}]}],
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        output = self.processor.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Parse [x1, y1, x2, y2]
        match = re.search(r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]', output)
        return [int(match.group(i)) for i in range(1, 5)] if match else None

    def run(self):
        if not INPUT_DIR.exists():
            print(f"Error: Input directory {INPUT_DIR} does not exist.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tracks = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("track_")])
        inference_list = []
        
        print(f"Processing {len(tracks)} tracks from {INPUT_DIR}...")
        
        for track in tqdm(tracks, desc="Cropping Test Set"):
            frames = []
            for i in range(1, 6):
                found = next((track / f"lr-00{i}{ext}" for ext in ['.jpg', '.png', '.jpeg'] if (track / f"lr-00{i}{ext}").exists()), None)
                if found: frames.append(found)
            
            if len(frames) != 5:
                continue

            out_track_dir = OUTPUT_DIR / track.name
            out_track_dir.mkdir(parents=True, exist_ok=True)
            cropped_paths = []
            
            for img_path in frames:
                img = cv2.imread(str(img_path))
                if img is None: continue
                img_resized = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))
                bbox = self.detect_bbox(img_resized)
                
                if bbox:
                    h, w = img_resized.shape[:2]
                    # Map 0-1000 back to pixels
                    x1, y1, x2, y2 = [int(v * dim / 1000) for v, dim in zip(bbox, [w, h, w, h])]
                    p = 5 # minimal padding
                    cropped = img_resized[max(0, y1-p):min(h, y2+p), max(0, x1-p):min(w, x2+p)]
                    
                    out_path = out_track_dir / img_path.name
                    cv2.imwrite(str(out_path), cropped)
                    cropped_paths.append(str(out_path.absolute()))
            
            if len(cropped_paths) == 5:
                inference_list.append(f"{','.join(cropped_paths)}\tNONE")
        
        list_path = OUTPUT_DIR / "public_test_cropped_list.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(inference_list))
            
        print(f"\nSuccess! Cropped images saved to: {OUTPUT_DIR}")
        print(f"Inference list created: {list_path}")

if __name__ == "__main__":
    PublicTestProcessor().run()
