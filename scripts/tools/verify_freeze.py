import paddle
import yaml
import sys
from pathlib import Path

# Add PaddleOCR to path
PADDLE_OCR_PATH = Path(__file__).resolve().parent.parent / "PaddleOCR"
sys.path.append(str(PADDLE_OCR_PATH))

from ppocr.modeling.architectures import build_model

import argparse

def verify_freeze(config_path):
    print(f"Loading config from: {config_path}")
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Temporary fix for out_channels if missing in config but needed for build
    if 'Head' in config['Architecture']:
        head_config = config['Architecture']['Head']
        if head_config.get('name') == 'MultiHead':
             # Provide dummy out_channels_list for MultiHead
             head_config['out_channels_list'] = {
                 'CTCLabelDecode': 38, 
                 'NRTRLabelDecode': 38, 
                 'SARLabelDecode': 38
             }
        elif 'out_channels' not in head_config:
            head_config['out_channels'] = 38 # Default placeholder

    # Build model
    print("Building model...")
    try:
        model = build_model(config['Architecture'])
    except Exception as e:
        print(f"Failed to build model: {e}")
        return

    # Check frozen parameters
    frozen_count = 0
    trainable_count = 0
    total_params = 0

    print("\n" + "="*80)
    print("PARAMETER FREEZE STATUS")
    print("="*80)
    
    # Store lines to print them in order
    frozen_lines = []
    trainable_lines = []

    for name, param in model.named_parameters():
        total_params += 1
        # In Paddle, stop_gradient=True means frozen
        is_frozen = (param.stop_gradient == True) or (param.trainable == False)
        
        status = "❄️  FROZEN" if is_frozen else "🔥 TRAIN "
        line = f"{status:12} | {name}"
        
        if is_frozen:
            frozen_count += 1
            frozen_lines.append(line)
        else:
            trainable_count += 1
            trainable_lines.append(line)

    # Print a sample of frozen and trainable layers
    print("\n--- Sample FROZEN Layers (First 10) ---")
    for line in frozen_lines[:10]:
        print(line)
    if len(frozen_lines) > 10: print(f"... and {len(frozen_lines)-10} more.")

    print("\n--- Sample TRAINABLE Layers (First 20) ---")
    for line in trainable_lines[:20]:
        print(line)
    if len(trainable_lines) > 20: print(f"... and {len(trainable_lines)-20} more.")

    print("\n" + "="*80)
    print(f"Summary for: {config_path.name}")
    print(f"  Total parameters: {total_params}")
    print(f"  Frozen: {frozen_count} ({100*frozen_count/total_params:.1f}%)")
    print(f"  Trainable: {trainable_count} ({100*trainable_count/total_params:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify frozen layers in PaddleOCR model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    args = parser.parse_args()
    
    verify_freeze(args.config)
