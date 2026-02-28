#!/usr/bin/env python3
"""
Script to analyze PP-OCRv5 pretrained model architecture compatibility.
"""

import argparse
import sys
from pathlib import Path
import yaml
import paddle

# Add PaddleOCR to path
PADDLE_OCR_PATH = Path(__file__).resolve().parent.parent / 'PaddleOCR'
sys.path.insert(0, str(PADDLE_OCR_PATH))

try:
    from ppocr.modeling.architectures import build_model
except ImportError:
    print(f"Error: Could not import ppocr from {PADDLE_OCR_PATH}")
    sys.exit(1)


def analyze_weights(pretrained_path):
    """Loads and lists components of the pretrained model weights."""
    print(f"Analyzing: {pretrained_path}")
    
    if not Path(pretrained_path).exists():
        print(f"Error: File not found: {pretrained_path}")
        return {}

    try:
        params = paddle.load(pretrained_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return {}

    print(f"Total parameters: {len(params)}")

    components = {}
    for key in params.keys():
        comp = key.split('.')[0]
        components.setdefault(comp, []).append(key)

    print("\nComponents found:")
    for comp, keys in sorted(components.items()):
        print(f"  - {comp}: {len(keys)} params")

    return params


def check_compatibility(config_path, pretrained_path):
    """Checks compatibility between a YAML config and pretrained weights."""
    print(f"\nChecking compatibility: Config={config_path} vs Pretrained={pretrained_path}")

    if not Path(config_path).exists():
        print(f"Error: Config not found: {config_path}")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Temporary fix for out_channels if missing in config but needed for build
    if 'Head' in config['Architecture']:
        head_config = config['Architecture']['Head']
        if head_config.get('name') == 'MultiHead':
             # Provide dummy out_channels_list for MultiHead e.g. {'CTCHead': 6625, 'NRTRHead': 6625}
             # We use a large number or try to infer, but just for model building, any int is fine usually
             # unless the shape check fails later, which is what we want to see.
             # Actually, MultiHead wrapper needs it passed.
             # Keys must match the Label Decode class names used in MultiHead
             head_config['out_channels_list'] = {
                 'CTCLabelDecode': 38, 
                 'NRTRLabelDecode': 38, 
                 'SARLabelDecode': 38
             }
        elif 'out_channels' not in head_config:
            head_config['out_channels'] = 38 # Default placeholder to avoid build errors

    try:
        model = build_model(config['Architecture'])
        model_state = model.state_dict()
    except Exception as e:
        print(f"Error building model from config: {e}")
        return False

    try:
        pretrained = paddle.load(pretrained_path)
    except Exception:
        return False
    
    exact_matches = []
    shape_mismatches = []
    missing_in_pretrained = []
    missing_in_model = []

    for key, shape in ((k, v.shape) for k, v in model_state.items()):
        if key not in pretrained:
            missing_in_pretrained.append(key)
            continue
            
        pre_shape = pretrained[key].shape
        if list(shape) == list(pre_shape):
            exact_matches.append(key)
        else:
            shape_mismatches.append((key, shape, pre_shape))

    for key in pretrained.keys():
        if key not in model_state:
            missing_in_model.append(key)

    total_model_params = len(model_state)
    match_rate = len(exact_matches) / total_model_params * 100 if total_model_params else 0

    print("\nCompatibility Results:")
    print(f"  Matches: {len(exact_matches)} ({match_rate:.1f}%)")
    print(f"  Shape Mismatches: {len(shape_mismatches)}")
    print(f"  Missing in Pretrained: {len(missing_in_pretrained)}")
    print(f"  Extra in Pretrained: {len(missing_in_model)}")

    if shape_mismatches:
        print("\n  [!] Shape Mismatches (First 10):")
        for key, m_shape, p_shape in shape_mismatches[:10]:
            print(f"    - {key}: Model{m_shape} != Pretrained{p_shape}")

    return len(exact_matches) > 0


def main():
    parser = argparse.ArgumentParser(description="Check PP-OCR model compatibility.")
    parser.add_argument("--pretrained", required=True, help="Path to .pdparams file")
    parser.add_argument("--config", required=True, help="Path to .yml config file")
    args = parser.parse_args()

    analyze_weights(args.pretrained)
    check_compatibility(args.config, args.pretrained)


if __name__ == "__main__":
    main()
