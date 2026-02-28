import paddle
import os
import sys
from pathlib import Path

def fix_sr_weights(input_path, output_path, algo="gestalt"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return False
    
    print(f"[*] Loading weights from {input_path}...")
    params = paddle.load(input_path)
    new_params = {}
    
    # Prefix mapping logic
    # In official SR checkpoints:
    # - SR backbone (TSRN/TBSRN) is prefixed with 'transform.'
    # - Guidance recognizer (Transformer) is prefixed with 'backbone.'
    # In PaddleOCR codebase (BaseModel + Transform):
    # - Everything is under 'transform.'
    # - Recognizer is 'transform.r34_transformer' (TSRN) or 'transform.transformer' (TBSRN)
    
    transformer_mapped_name = "r34_transformer" if algo == "gestalt" else "transformer"
    
    print(f"[*] Mapping prefixes for algorithm: {algo}")
    count_mapped = 0
    count_kept = 0
    
    for k, v in params.items():
        if k.startswith("backbone."):
            new_key = k.replace("backbone.", f"transform.{transformer_mapped_name}.")
            new_params[new_key] = v
            count_mapped += 1
        else:
            new_params[k] = v
            count_kept += 1
            
    print(f"[+] Mapped {count_mapped} backbone keys to transform branch.")
    print(f"[+] Kept {count_kept} transform keys.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paddle.save(new_params, output_path)
    print(f"[+] Fixed weights saved to {output_path}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--algo", type=str, default="gestalt")
    args = parser.parse_args()
    
    fix_sr_weights(args.input, args.output, args.algo)
