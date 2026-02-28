# Feature 3 — Selective Freeze Strategy (SFM vs PSM)

### The Problem

The Gestalt SR model (TSRN) consists of two conceptually distinct modules:

| Module | Description | Role during training |
|:---|:---|:---|
| **PSM** (Pixel-aware Super-resolution Module) | STN + TSRN blocks + Upsampler | Transforms LR → HR pixels |
| **SFM** (Stroke-aware Functional Module) | `r34_transformer` (Transformer-based recognizer) | Provides text recognition signal for `StrokeFocusLoss` |

The SFM is a pretrained English text recognizer frozen during SR training. Its purpose is purely **supervisory** — it computes `attention_loss` by comparing what the recognizer "sees" in the SR output vs the HR ground truth. If the recognizer changes during training, this reference signal becomes unstable, making the loss diverge.

**PaddleOCR's default TSRN freezes the SFM incorrectly** (or inconsistently across versions). We need to guarantee the freeze in the source.

### The Fix

**File:** `PaddleOCR/ppocr/modeling/transforms/tsrn.py`

The `r34_transformer` (SFM for Gestalt) is hard-frozen during `__init__`:

```python
# tsrn.py — TSRN.__init__
self.r34_transformer = Transformer()   # SFM: pretrained English recognizer

# Freeze ALL parameters of the recognizer immediately after construction
for param in self.r34_transformer.parameters():
    param.trainable = False   # excluded from optimizer updates
    # Note: stop_gradient is implicitly True when trainable=False in PaddlePaddle
```

The equivalent for the Telescope model (`tbsrn.py`) freezes `self.transformer` instead:

```python
# tbsrn.py — TBSRN.__init__
self.transformer = Transformer()
for param in self.transformer.parameters():
    param.trainable = False
```

**What remains trainable (PSM):**
- `self.stn_head` — the STN that predicts warping control points
- `self.tps` — thin-plate spline transformer
- `self.block1` through `self.block{srb_nums+3}` — the TSRN residual SR blocks
- The upsampler (`UpsampleBLock`)

**Freeze verification in `forward`:**

```python
# tsrn.py — TSRN.forward
if self.training:
    hr_img = x[1]
    length = x[2]
    input_tensor = x[3]

    # SFM runs in inference mode (no grad, frozen weights)
    sr_pred, word_attention_map_pred, _ = self.r34_transformer(
        sr_img, length, input_tensor
    )
    hr_pred, word_attention_map_gt, _ = self.r34_transformer(
        hr_img, length, input_tensor
    )
    # These outputs feed directly into StrokeFocusLoss
    output["sr_pred"] = sr_pred
    output["hr_pred"] = hr_pred
    output["word_attention_map_pred"] = word_attention_map_pred
    output["word_attention_map_gt"]   = word_attention_map_gt
```

The SFM is called during training only to compute the auxiliary loss — it never updates its own weights.

### Why This Design Matters

The `StrokeFocusLoss` penalizes the SR model when its output looks different from the HR image *as perceived by the recognizer*. By keeping the recognizer frozen, the loss gradient flows only through the PSM (the SR backbone), driving it to produce images that look "readable" to the text recognizer — not just pixel-accurate to HR.

---

