# Feature 1 — Transfer Learning via Backbone Stage Freezing

### The Problem

When fine-tuning a large pretrained backbone (PP-HGNetV2-B4) on a small plate dataset (~50K images), the entire network tends to overfit quickly. The early layers (stem, stage 1) encode generic low-level features (edges, textures) that are already optimal from ImageNet pretraining — retraining them wastes compute and risks catastrophic forgetting.

**PaddleOCR's default `PPHGNetV2` does not support selective layer freezing.** Passing `frozen_stages=2` to the YAML config does nothing — the parameter is silently discarded.

### The Fix

**Files modified:**
- `PaddleOCR/ppocr/modeling/backbones/rec_pphgnetv2.py`
- `PaddleOCR/ppocr/modeling/backbones/rec_lcnetv3.py`

We add two methods to `PPHGNetV2.__init__`: a `frozen_stages` parameter and a `_freeze_stages()` call at the end of `__init__`. This ensures freezing happens immediately at model construction, before any optimizer is built.

```python
# rec_pphgnetv2.py — PPHGNetV2.__init__
def __init__(
    self,
    ...,
    frozen_stages=-1,   # -1 = freeze nothing; 0 = freeze stem+stage0; 1 = +stage1; etc.
    **kwargs,
):
    ...
    self.frozen_stages = frozen_stages
    self._init_weights()
    self._freeze_stages()   # <-- called here, after all layers are defined
```

The `_freeze_stages` method iterates over the stem and numbered stages, stopping at `stop_gradient`:

```python
def _freeze_parameters(self, m):
    for param in m.parameters():
        param.trainable = False
        param.stop_gradient = True

def _freeze_stages(self):
    if self.frozen_stages < 0:
        return

    # Freeze stem (feature extraction entry point)
    if hasattr(self, 'stem'):
        print(f"[FREEZE] Freezing stem")
        self._freeze_parameters(self.stem)

    # Freeze stages 0 through frozen_stages (inclusive)
    for i in range(self.frozen_stages + 1):
        if i < len(self.stages):   # guard: safe in temporal mode where stages list is shorter
            print(f"[FREEZE] Freezing stage {i}")
            self._freeze_parameters(self.stages[i])
```

> **Why `if i < len(self.stages)`?** In temporal fusion mode (`use_temporal=True`), stages 3-4 are still instantiated but the logical `self.stages` may have fewer active entries. The guard prevents `IndexError` in this edge case.

### LCNetV3 Variant

`rec_lcnetv3.py` receives the same `frozen_stages` parameter, but its freezing logic is a cascade of threshold checks rather than an index loop — because LCNetV3's stage hierarchy is different:

```python
# rec_lcnetv3.py — _freeze_stages
def _freeze_stages(self):
    if self.frozen_stages <= 0:
        return                     # freeze nothing
    if self.frozen_stages >= 2:
        self._freeze_parameters(self.stage2)   # freeze Stage 2
    if self.frozen_stages >= 3:
        self._freeze_parameters(self.stage3)   # freeze Stage 3
    if self.frozen_stages >= 4:
        self._freeze_parameters(self.stage4)   # freeze Stage 4
    if self.frozen_stages >= 5:
        self._freeze_parameters(self.stage5)   # freeze Stage 5
```

### Usage in Config

```yaml
# configs/rec_v5_template.yaml
Backbone:
  name: PPHGNetV2_B4
  frozen_stages: 1    # freeze stem + stage0 + stage1; train stage2+ and head
```

To verify freezing is applied correctly, run:

```bash
uv run python scripts/tools/verify_freeze.py --config configs/ensemble_v5/rec/rec_v5_model_1.yml
# Expected output: trainable vs frozen parameter counts per layer
```

---

