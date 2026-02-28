# Feature 2 — STN Linear Layer Fix

## The Problem

The Spatial Transformer Network (STN) inside TSRN processes the input image through 5 consecutive Conv + MaxPool layers, then flattens the result and passes it through a fully connected layer `stn_fc1` to predict thin-plate spline control points.

The original code hardcodes the input size of `stn_fc1` as **512**. This value was calibrated for the default `32×128` training resolution. To understand why it breaks for `48×320`, trace the `stn_convnet` architecture explicitly:

```
Layer                        Channels    H    W
────────────────────────────────────────────────
Input                            3      48   320
conv3x3_block(3, 32)            32      48   320
MaxPool2D(2, stride=2)          32      24   160
conv3x3_block(32, 64)           64      24   160
MaxPool2D(2, stride=2)          64      12    80
conv3x3_block(64, 128)         128      12    80
MaxPool2D(2, stride=2)         128       6    40
conv3x3_block(128, 256)        256       6    40
MaxPool2D(2, stride=2)         256       3    20
conv3x3_block(256, 256)        256       3    20
MaxPool2D(2, stride=2)         256       1    10  ← floor(3/2)=1, floor(20/2)=10
conv3x3_block(256, 256)        256       1    10
────────────────────────────────────────────────
Flattened: 256 × 1 × 10 = 2560
```

With the original hardcoded `512`, the `nn.Linear(512, ...)` call fails immediately with a shape mismatch error when using `48×320` inputs:

```
ValueError: (ShapeError) The two dimensions of the input are expected to be equal.
Received X's dimensions: [B, 2560], Linear's weight: [512, 512]
```

## The Fix

**File modified:** `PaddleOCR/ppocr/modeling/transforms/stn.py`

Change the `stn_fc1` input dimension from `512` to `2560`:

```python
# stn.py — STN.__init__
# Before (calibrated for 16×128 STN input — crashes for 48×320):
# nn.Linear(512, 512, ...)

# After (calibrated for 48×320):
self.stn_fc1 = nn.Sequential(
    nn.Linear(
        2560,   # 256 channels × 1 height × 10 width = 2560 for 48×320 input
        512,
        weight_attr=nn.initializer.Normal(0, 0.001),
        bias_attr=nn.initializer.Constant(0),
    ),
    nn.BatchNorm1D(512),
    nn.ReLU(),
)
```

> **Trade-off — Hardcoded value:** This dimension is not derived dynamically from `imgH`/`imgW`. `stn.py` is now tightly coupled to `48×320` inputs. If you ever change the target resolution, recompute manually:
>
> ```
> flattened_size = 256 × floor(imgH / 32) × floor(imgW / 32)
> ```
>
> Examples:
> - `48×320`: `256 × 1 × 10 = 2560` ← current
> - `32×128`: `256 × 1 × 4 = 1024`
> - `64×256`: `256 × 2 × 8 = 4096`

---
