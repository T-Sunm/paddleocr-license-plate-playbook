import argparse
import sys
from pathlib import Path
from datetime import datetime

from plate_ocr.paths import PROJECT_ROOT, CONFIGS_DIR, OUTPUT_DIR
from plate_ocr.utils.file_utils import ensure_dir


def generate_configs(args):
    """
    Generate single config for Teacher (Step 1) or Distillation (Step 2).
    Follows "Low Risk, High Efficiency" plan: No ensemble.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if args.mode == "teacher":
        template_path = CONFIGS_DIR / "ensemble" / "rec" / "rec_v5_template.yaml"
        config_output_dir = CONFIGS_DIR / "teacher"
        output_dir = OUTPUT_DIR / f"teacher_v5_{timestamp}"
        label_train = "train_teacher_hr.txt"
        label_val = "val_teacher_hr.txt"
    else:
        template_path = CONFIGS_DIR / "distill" / "rec_v5_distill_template.yaml"
        config_output_dir = CONFIGS_DIR / "distill"
        output_dir = OUTPUT_DIR / f"distill_v5_{timestamp}"
        label_train = "train_student_pair.txt"
        label_val = "val_student_pair.txt"

    ensure_dir(config_output_dir)
    ensure_dir(output_dir)

    distill_data_dir = PROJECT_ROOT / "data" / "processed_rec_distill"

    with open(template_path, 'r') as f:
        template_content = f.read()

    config_content = template_content
    config_content = config_content.replace("REPLACE_OUTPUT_DIR", str(output_dir))
    config_content = config_content.replace("REPLACE_DATA_DIR", str(distill_data_dir))
    config_content = config_content.replace("train.txt", label_train)
    config_content = config_content.replace("val.txt", label_val)

    dict_path = PROJECT_ROOT / "data" / "dict" / "plate_dict.txt"
    config_content = config_content.replace("REPLACE_DICT_PATH", str(dict_path))

    if args.mode == "distill":
        if not args.teacher_path:
            print("Error: --teacher-path is required for distill mode")
            sys.exit(1)
        config_content = config_content.replace("REPLACE_TEACHER_PRETRAINED", str(args.teacher_path))
        config_content = config_content.replace("REPLACE_STUDENT_PRETRAINED", str(args.student_pretrained))

    config_filename = f"rec_v5_{args.mode}.yml"
    config_path = config_output_dir / config_filename

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"[+] Generated {args.mode.upper()} config: {config_path}")
    print(f"    -> Output directory: {output_dir}")
    if args.mode == "distill":
        print(f"    -> Teacher used: {args.teacher_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Single Teacher/Distill config")
    parser.add_argument("--mode", type=str, choices=["teacher", "distill"], required=True)
    parser.add_argument("--teacher-path", type=str, help="Path to Step 1 best_accuracy model (required for distill mode)")
    parser.add_argument("--student-pretrained", type=str,
                        default=str(PROJECT_ROOT / "weights" / "pretrained" / "ocr" / "paddleocr_v5_rec" / "PP-OCRv5_server_rec_pretrained"))
    args = parser.parse_args()

    generate_configs(args)
