#!/usr/bin/env python3
import argparse
import csv
import sys
import yaml
import zipfile
import numpy as np
import paddle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from plate_ocr.paths import PROJECT_ROOT, PADDLE_OCR_ROOT, OUTPUT_DIR

sys.path.append(str(PADDLE_OCR_ROOT))

from plate_ocr.utils.metrics import norm_edit_dis
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.data import create_operators, transform


class MultiframeInferer:
    def __init__(self, config_path, checkpoint_path=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        global_config = self.config["Global"]
        self.config["Architecture"]["Backbone"]["use_temporal"] = True

        self.post_process_class = build_post_process(self.config["PostProcess"], global_config)

        char_num = len(getattr(self.post_process_class, "character"))
        if self.config["Architecture"]["Head"]["name"] == "MultiHead":
            self.config["Architecture"]["Head"]["out_channels_list"] = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 3
            }
        else:
            self.config["Architecture"]["Head"]["out_channels"] = char_num

        self.model = build_model(self.config["Architecture"])

        if checkpoint_path:
            self.config["Global"]["checkpoints"] = checkpoint_path
        load_model(self.config, self.model)
        self.model.eval()

        transforms_config = []
        for op in self.config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms_config.append(op)

        global_config["infer_mode"] = True
        self.ops = create_operators(transforms_config, global_config)

    def predict(self, img_paths):
        img_list = []
        for p in img_paths:
            with open(p, "rb") as f:
                img_list.append(f.read())

        batch = transform({"image": img_list}, self.ops)
        images = paddle.to_tensor(np.expand_dims(batch[0], axis=0))

        with paddle.no_grad():
            preds = self.model(images)

        res = self.post_process_class(preds)
        if res and len(res[0]) >= 2:
            return res[0][0], float(res[0][1])
        return "", 0.0


def main():
    parser = argparse.ArgumentParser(description="Multiframe Recognition Inference")
    parser.add_argument("--track_dir", type=str, help="Directory containing frames")
    parser.add_argument("--label_file", type=str, help="Path to label file for batch inference")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--config", type=str,
                        default="configs/multiframe/rec_v5_multiframe.yml",
                        help="Path to config file (relative to project root)")
    parser.add_argument("--output", type=str, help="Path to save result CSV")
    parser.add_argument("--zip", type=str, help="Path to save result ZIP")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    print(f"Initializing Inferer: {config_path}")
    inferer = MultiframeInferer(config_path, args.checkpoint)

    if args.track_dir:
        track_path = Path(args.track_dir)
        imgs = []
        for i in range(1, 6):
            found_path = next(
                (track_path / f"lr-00{i}{ext}"
                 for ext in ['.png', '.jpg', '.jpeg']
                 if (track_path / f"lr-00{i}{ext}").exists()),
                None
            )
            if not found_path:
                print(f"Error: Frame {i} not found in {args.track_dir}")
                return
            imgs.append(str(found_path))

        text, score = inferer.predict(imgs)
        print(f"{'-'*50}\nTrack: {args.track_dir}\nPrediction: {text}\nConfidence: {score:.4f}\n{'-'*50}")

    elif args.label_file:
        label_file = Path(args.label_file)
        if not label_file.exists():
            print(f"Error: {label_file} not found")
            return

        with open(label_file, 'r') as f:
            lines = f.readlines()

        results = []
        for line in tqdm(lines, desc="Processing batch"):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            img_paths_str, gt = parts
            img_paths = img_paths_str.split(',')

            try:
                pred, score = inferer.predict(img_paths)
                results.append({
                    "track_id": Path(img_paths[0]).parent.name,
                    "paths": img_paths_str,
                    "gt": gt,
                    "pred": pred,
                    "score": score,
                    "match": int(pred == gt),
                    "ned": norm_edit_dis(pred, gt)
                })
            except Exception as e:
                print(f"Error processing {img_paths_str}: {e}")

        if not results:
            return

        acc = sum(r["match"] for r in results) / len(results)
        mean_ned = sum(r["ned"] for r in results) / len(results)

        print(f"\n{'='*50}\nMetrics: {label_file.name}\nSamples: {len(results)}\nAccuracy: {acc:.4f}\nMean NED: {mean_ned:.4f}\n{'='*50}")

        out_csv = Path(args.output) if args.output else OUTPUT_DIR / f"eval_mf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved CSV: {out_csv}")

        if args.zip:
            zip_path = Path(args.zip)
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(f"{r['track_id']},{r['pred']};{r['score']:.4f}" for r in results)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("predictions.txt", content)
            print(f"Saved ZIP: {zip_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
