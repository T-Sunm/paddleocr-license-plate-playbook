import os
import argparse
import requests
from pathlib import Path

from plate_ocr.paths import WEIGHTS_DIR

DET_SAVE_DIR = WEIGHTS_DIR / "pretrained" / "ocr" / "paddleocr_v5_det"
REC_SAVE_DIR = WEIGHTS_DIR / "pretrained" / "ocr" / "paddleocr_v5_rec"


def download_model(model_name: str, model_url: str, save_dir: Path, hf_repo_id: str = None) -> Path:
    """Generic model downloader from Baidu BOS with an optional HuggingFace fallback."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] Downloading {model_name}...")
    print(f"    Target: {save_dir}")

    filename = model_url.split('/')[-1]
    model_path = save_dir / filename

    try:
        print(f"[*] Downloading from Baidu BOS...")
        if model_path.exists():
            print(f"[!] Model already exists at {model_path}, skipping download.")
            return save_dir

        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"[*] Progress: {percent:.1f}% ({downloaded_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end='\r')

        print("\n[+] Download completed!")
        print(f"[+] Model saved to: {model_path}")
        return save_dir

    except Exception as e:
        print(f"\n[!] Baidu BOS Error: {e}")

        if hf_repo_id:
            print("[*] Trying alternative method with HuggingFace Hub...")
            try:
                from huggingface_hub import hf_hub_download

                files_to_download = [
                    "inference.pdiparams",
                    "inference.pdiparams.info",
                    "inference.pdmodel",
                    "inference.json"
                ]

                print(f"[*] Downloading from HuggingFace: {hf_repo_id}")
                for hf_filename in files_to_download:
                    try:
                        print(f"[*] Downloading {hf_filename}...")
                        hf_hub_download(
                            repo_id=hf_repo_id,
                            filename=hf_filename,
                            cache_dir=save_dir,
                            local_dir=save_dir / "inference",
                            local_dir_use_symlinks=False
                        )
                        print(f"[+] Downloaded: {hf_filename}")
                    except Exception as file_error:
                        print(f"[!] Failed to download {hf_filename}: {file_error}")

                inference_dir = save_dir / "inference"
                if inference_dir.exists():
                    print(f"[+] Model extracted to: {inference_dir}")
                    return inference_dir

            except Exception as hf_error:
                print(f"[!] HuggingFace Hub Error: {hf_error}")

        print("\n[!] All methods failed. Please try:")
        print("    1. Install paddleocr: pip install paddleocr")
        if hf_repo_id:
            print("    2. Let PaddleOCR auto-download it by running default scripts")
        else:
            print(f"    2. Try manually downloading from: {model_url}")
        return None


def download_pp_ocrv5_detection_model(save_dir=DET_SAVE_DIR):
    """Download PP-OCRv5 server detection model."""
    return download_model(
        model_name="PP-OCRv5 server detection model",
        model_url="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams",
        save_dir=save_dir,
        hf_repo_id="PaddlePaddle/PP-OCRv5_server_det"
    )


def download_pp_ocrv5_recognition_model(save_dir=REC_SAVE_DIR):
    """Download PP-OCRv5 server recognition model."""
    return download_model(
        model_name="PP-OCRv5 server recognition model",
        model_url="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams",
        save_dir=save_dir,
        hf_repo_id=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PP-OCRv5 pretrained models")
    parser.add_argument("--mode", type=str, choices=["det", "rec", "both"], default="both",
                        help="Which model to download: det, rec, or both")
    args = parser.parse_args()

    success = True

    if args.mode in ["det", "both"]:
        print("\n" + "="*80)
        print("DOWNLOADING DETECTION MODEL")
        print("="*80)
        det_dir = download_pp_ocrv5_detection_model()
        if det_dir:
            print(f"\n[✓] Detection model ready at: {det_dir}")
        else:
            success = False

    if args.mode in ["rec", "both"]:
        print("\n" + "="*80)
        print("DOWNLOADING RECOGNITION MODEL")
        print("="*80)
        rec_dir = download_pp_ocrv5_recognition_model()
        if rec_dir:
            print(f"\n[✓] Recognition model ready at: {rec_dir}")
        else:
            success = False

    if success:
        print("\n" + "="*80)
        print("[✓] ALL MODELS DOWNLOADED SUCCESSFULLY")
        print("="*80)
