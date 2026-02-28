# Feature 4 — Pretrained Weight Prefix Remapping

### The Problem

PaddleOCR's official released SR checkpoints (`sr_tsrn_transformer_strock_train`) use a **different key naming convention** than what PaddleOCR's `BaseModel` wrapper expects at inference/fine-tuning time:

| What | Official checkpoint keys | PaddleOCR BaseModel expects |
|:---|:---|:---|
| SR backbone (TSRN) | `transform.*` | `transform.*` ✓ |
| Recognizer (SFM) | `backbone.*` | `transform.r34_transformer.*` ✗ |

When you directly load the official weights, PaddleOCR prints:

```
[WARNING] There are 47 out of 47 params of backbone not used in the model.
```

This means the SFM (which provides the supervision signal through `StrokeFocusLoss`) starts from random initialization instead of the pretrained English recognizer. Training will still proceed, but the StrokeFocusLoss will be uninformative until the SFM re-learns from scratch — wasting many epochs.

### The Fix

**File:** `scripts/tools/fix_sr_pretrained.py`

A one-time remapping script that reads the official checkpoint, renames any key starting with `backbone.` to `transform.{recognizer_name}.`, and saves a fixed checkpoint:

```python
# fix_sr_pretrained.py

def fix_sr_weights(input_path, output_path, algo="gestalt"):
    params = paddle.load(input_path)
    new_params = {}

    # Mapping logic:
    # Official checkpoint: "backbone.xxx"  → layers under a standalone recognizer
    # PaddleOCR BaseModel: "transform.r34_transformer.xxx" (Gestalt)
    #                   or "transform.transformer.xxx"     (Telescope)
    transformer_mapped_name = "r34_transformer" if algo == "gestalt" else "transformer"

    count_mapped = 0
    for k, v in params.items():
        if k.startswith("backbone."):
            # Remap: backbone.xxx → transform.<recognizer>.xxx
            new_key = k.replace("backbone.", f"transform.{transformer_mapped_name}.")
            new_params[new_key] = v
            count_mapped += 1
        else:
            # transform.xxx keys are already correct
            new_params[k] = v

    paddle.save(new_params, output_path)
    print(f"[+] Remapped {count_mapped} backbone keys → transform.{transformer_mapped_name}.*")
```

**Usage:**

```bash
# For Gestalt (TSRN):
python scripts/tools/fix_sr_pretrained.py \
    --input  weights/pretrained/sr/sr_tsrn_transformer_strock_train/best_accuracy.pdparams \
    --output weights/pretrained/sr/sr_tsrn_transformer_strock_train/best_accuracy_fixed.pdparams \
    --algo   gestalt

# For Telescope (TBSRN):
python scripts/tools/fix_sr_pretrained.py \
    --input  weights/pretrained/sr/sr_tbsrn_transformer_strock_train/best_accuracy.pdparams \
    --output weights/pretrained/sr/sr_tbsrn_transformer_strock_train/best_accuracy_fixed.pdparams \
    --algo   telescope
```

After running, configure the fixed checkpoint in your YAML:

```yaml
# configs/sr/sr_gestalt_plate.yml
Global:
  pretrained_model: weights/pretrained/sr/sr_tsrn_transformer_strock_train/best_accuracy_fixed
  #                                                                                      ↑
  #                               Always use _fixed, never the raw downloaded checkpoint
```

> **Verify the fix:** After loading, PaddleOCR should print **zero "not used" warnings**. Any remaining warnings indicate additional key mismatches that need investigation.

---

