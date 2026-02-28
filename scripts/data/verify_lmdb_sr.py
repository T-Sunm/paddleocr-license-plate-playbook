#!/usr/bin/env python3
"""
Verify LMDB dataset for SR training.

This script checks the integrity and format of LMDB datasets built for
PaddleOCR SR training (Telescope/Gestalt).
"""

import argparse
import lmdb
import io
from PIL import Image


def verify_lmdb(lmdb_path, num_samples_to_check=5):
    """
    Verify LMDB dataset format and integrity.
    
    Args:
        lmdb_path: Path to LMDB directory
        num_samples_to_check: Number of samples to display details for
    """
    print(f"\n{'='*60}")
    print(f"Verifying LMDB: {lmdb_path}")
    print(f"{'='*60}\n")
    
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
    except Exception as e:
        print(f"❌ Failed to open LMDB: {e}")
        return False
    
    with env.begin() as txn:
        # Check num-samples
        num_samples_bytes = txn.get(b'num-samples')
        if num_samples_bytes is None:
            print("❌ Missing 'num-samples' key")
            env.close()
            return False
        
        num_samples = int(num_samples_bytes.decode('utf-8'))
        print(f"✓ Total samples: {num_samples}")
        
        # Check sample keys
        missing_keys = []
        invalid_images = []
        
        for idx in range(1, min(num_samples + 1, num_samples_to_check + 1)):
            label_key = f'label-{idx:09d}'.encode()
            hr_key = f'image_hr-{idx:09d}'.encode()
            lr_key = f'image_lr-{idx:09d}'.encode()
            
            # Check label
            label_bytes = txn.get(label_key)
            if label_bytes is None:
                missing_keys.append(label_key.decode())
                continue
            label = label_bytes.decode('utf-8')
            
            # Check HR image
            hr_bytes = txn.get(hr_key)
            if hr_bytes is None:
                missing_keys.append(hr_key.decode())
                continue
            
            try:
                hr_img = Image.open(io.BytesIO(hr_bytes)).convert('RGB')
                hr_size = hr_img.size
            except Exception as e:
                invalid_images.append((hr_key.decode(), str(e)))
                continue
            
            # Check LR image
            lr_bytes = txn.get(lr_key)
            if lr_bytes is None:
                missing_keys.append(lr_key.decode())
                continue
            
            try:
                lr_img = Image.open(io.BytesIO(lr_bytes)).convert('RGB')
                lr_size = lr_img.size
            except Exception as e:
                invalid_images.append((lr_key.decode(), str(e)))
                continue
            
            # Display sample info
            print(f"\nSample {idx}:")
            print(f"  Label: {label}")
            print(f"  HR size: {hr_size[0]}x{hr_size[1]} (W×H)")
            print(f"  LR size: {lr_size[0]}x{lr_size[1]} (W×H)")
            
            # Calculate scale ratio
            if lr_size[0] > 0 and lr_size[1] > 0:
                scale_w = hr_size[0] / lr_size[0]
                scale_h = hr_size[1] / lr_size[1]
                print(f"  Scale ratio: {scale_w:.2f}x (W), {scale_h:.2f}x (H)")
        
        # Summary
        print(f"\n{'='*60}")
        print("Verification Summary")
        print(f"{'='*60}")
        
        if missing_keys:
            print(f"❌ Missing keys: {len(missing_keys)}")
            for key in missing_keys[:10]:
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more")
        else:
            print(f"✓ All checked keys present")
        
        if invalid_images:
            print(f"❌ Invalid images: {len(invalid_images)}")
            for key, error in invalid_images[:10]:
                print(f"   - {key}: {error}")
            if len(invalid_images) > 10:
                print(f"   ... and {len(invalid_images) - 10} more")
        else:
            print(f"✓ All checked images valid")
        
        print()
    
    env.close()
    return len(missing_keys) == 0 and len(invalid_images) == 0


def main():
    parser = argparse.ArgumentParser(description='Verify LMDB dataset for SR training')
    parser.add_argument(
        '--lmdb_dir',
        type=str,
        required=True,
        help='Path to LMDB directory (e.g., .../lmdb_sr/train_lmdb)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to check in detail (default: 5)'
    )
    
    args = parser.parse_args()
    
    success = verify_lmdb(args.lmdb_dir, args.num_samples)
    
    if success:
        print("✓ LMDB verification passed!")
    else:
        print("❌ LMDB verification failed!")
        exit(1)


if __name__ == '__main__':
    main()
