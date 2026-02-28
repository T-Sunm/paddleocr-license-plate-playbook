# Feature 1 — Input Size Alignment (48×320)

### The Problem

The standard PaddleOCR SR model was trained and validated on a `32×128` image size — the canonical resolution for general English text recognition. Our OCR recognition model, however, expects a `48×320` input (taller and wider to better capture the aspect ratio of Vietnamese/international license plates).

If the SR model outputs `32×128` images, feeding them into the OCR model requires an upsampling step that introduces additional blur and artifacts, negating the SR gain entirely. **The SR output must exactly match the OCR input.**

### The Fix

**Files modified:** `configs/sr/sr_gestalt_plate.yml` (config), `configs/sr/sr_telescope_plate.yml`

The target dimensions are set in `SRResize` inside the config. The `down_sample_scale: 1` setting is equally important — it tells the model to treat the input as full-resolution and enhance its perceptual quality rather than upscaling by a fixed factor:

```yaml
Train:
  dataset:
    transforms:
      - SRResize:
          imgH: 48             # ← height aligned to OCR input
          imgW: 320            # ← width aligned to OCR input
          down_sample_scale: 1 # ← enhancement mode, not upscaling mode
```

The same dimensions are set for `Eval` to ensure validation metrics are computed at the correct scale:

```yaml
Eval:
  dataset:
    transforms:
      - SRResize:
          imgH: 48
          imgW: 320
          down_sample_scale: 1
```

And inside the TSRN architecture block — `width` and `height` control the STN's internal warping grid:

```yaml
Architecture:
  model_type: sr
  algorithm: Gestalt
  Transform:
    name: TSRN
    STN: True
    width: 320     # ← must match imgW
    height: 48     # ← must match imgH
    scale_factor: 1
```

> **Why `scale_factor: 1`?** The original TSRN uses `scale_factor: 2` to double resolution. In our setup, both LR and HR crops are already at the same pixel dimensions — the task is quality enhancement (deblurring, sharpening) rather than super-resolution in the classical sense. Setting `scale_factor: 1` disables the upsampling block.

---
