import argparse
import sys
from pathlib import Path
from datetime import datetime

from plate_ocr.paths import PROJECT_ROOT, PADDLE_OCR_ROOT, CONFIGS_DIR, OUTPUT_DIR
from plate_ocr.utils.file_utils import ensure_dir


def generate_configs(template_path, config_output_dir_base, model_output_dir_base, data_models_dir, mode="det", num_models=5):
    """Generate 5 specific config files from a template."""
    ensure_dir(config_output_dir_base)
    ensure_dir(model_output_dir_base)

    with open(template_path, 'r') as f:
        template_content = f.read()

    generated_files = []

    for i in range(1, num_models + 1):
        model_name = f"model_{i}"

        data_dir = data_models_dir / model_name
        save_model_dir = model_output_dir_base / model_name

        config_content = template_content.replace("REPLACE_OUTPUT_DIR", str(save_model_dir))
        config_content = config_content.replace("REPLACE_DATA_DIR", str(data_dir))

        if mode == "rec":
            dict_path = PROJECT_ROOT / "data" / "dict" / "plate_dict.txt"
            config_content = config_content.replace("REPLACE_DICT_PATH", str(dict_path))

        config_filename = f"{mode}_v5_{model_name}.yml"
        config_path = config_output_dir_base / config_filename

        with open(config_path, 'w') as f:
            f.write(config_content)

        generated_files.append(config_path)
        print(f"[+] Generated config: {config_path}")
        print(f"    -> Model output: {save_model_dir}")

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ensemble recognition configs")
    parser.add_argument("--template", type=str, default=None, help="Path to specific template file")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    mode = "rec"
    
    DATA_MODELS_DIR = PROJECT_ROOT / "data/models/paddleocr/rec_data"
    CONFIGS_OUTPUT_DIR = CONFIGS_DIR / "ensemble" / "rec"
    MODELS_OUTPUT_DIR = OUTPUT_DIR / f"ensemble_v5/rec_{timestamp}"

    default_template = "rec_v5_template.yaml"
    if args.template:
        TEMPLATE_PATH = Path(args.template) if Path(args.template).is_absolute() else CONFIGS_DIR / "ensemble" / mode / args.template
    else:
        TEMPLATE_PATH = CONFIGS_DIR / "ensemble" / mode / default_template

    ensure_dir(CONFIGS_OUTPUT_DIR)

    if not TEMPLATE_PATH.exists():
        print(f"Error: Template not found: {TEMPLATE_PATH}")
        sys.exit(1)

    print(f"[*] Generating {mode} configs from {TEMPLATE_PATH}...")
    print(f"[*] Data models dir: {DATA_MODELS_DIR}")
    print(f"[*] Configs output: {CONFIGS_OUTPUT_DIR}")
    print(f"[*] Models output: {MODELS_OUTPUT_DIR}")

    generate_configs(
        template_path=TEMPLATE_PATH,
        config_output_dir_base=CONFIGS_OUTPUT_DIR,
        model_output_dir_base=MODELS_OUTPUT_DIR,
        data_models_dir=DATA_MODELS_DIR,
        mode=mode
    )
    print("[✓] All configs generated.")
