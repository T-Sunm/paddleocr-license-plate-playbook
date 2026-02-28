import sys
from pathlib import Path
from datetime import datetime

import os
from plate_ocr.paths import CONFIGS_DIR, OUTPUT_DIR, PROJECT_ROOT

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def generate_config():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    import re

    template_path = CONFIGS_DIR / "multiframe" / "rec_multiframe.yml"
    output_dir = OUTPUT_DIR / f"multiframe_{timestamp}"
    data_dir = PROJECT_ROOT / "data/processed_rec_multiframe"
    dict_path = PROJECT_ROOT / "data/dict/plate_dict.txt"

    ensure_dir(output_dir)

    with open(template_path, 'r') as f:
        content = f.read()

    # Update paths in-place using regex
    content = re.sub(r'save_model_dir:.*', f'save_model_dir: {output_dir}', content)
    content = re.sub(r'character_dict_path:.*', f'character_dict_path: {dict_path}', content)
    content = re.sub(r'data_dir:.*', f'data_dir: {data_dir}', content)
    
    # Update label_file_list for train and val
    content = re.sub(r'(?<=label_file_list:\n)\s+- .*train.txt', f'      - {data_dir}/train.txt', content)
    content = re.sub(r'(?<=label_file_list:\n)\s+- .*val.txt', f'      - {data_dir}/val.txt', content)

    # Nếu còn tàn dư của chữ REPLACE_ thì cũng replace luôn để phòng hờ
    content = content.replace("REPLACE_DATA_DIR", str(data_dir))
    content = content.replace("REPLACE_DICT_PATH", str(dict_path))

    with open(template_path, 'w') as f:
        f.write(content)

    print(f"[+] Updated config in-place: {template_path}")
    print(f"    -> Next Output directory will be: {output_dir}")


if __name__ == "__main__":
    generate_config()
