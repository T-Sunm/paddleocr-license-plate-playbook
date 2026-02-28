# Feature 4 — Distillation on Asymmetric LR/HR Inputs (LRLP)

### The Problem

> [!NOTE]
> This approach is inspired by [One-stage Low-resolution Text Recognition with High-resolution Knowledge Transfer (2023)](https://arxiv.org/pdf/2308.02770).

Knowledge Distillation is a standard technique: a large, accurate Teacher guides a smaller Student by matching their output distributions. PaddleOCR ships with built-in distillation support via `DistillationModel` and `DistillationNRTRDMLLoss`.

However, these assume **Teacher and Student see the same input image**. In the LRLP (Low-Resolution License Plate) scenario, this is fundamentally wrong:

- **Teacher** was trained on high-resolution (HR) crops and must receive HR inputs to produce useful soft labels.
- **Student** will be deployed on low-resolution (LR) crops from video — it should only ever see LR inputs.

Feeding both the same image collapses the quality gap that distillation is supposed to bridge.

There are **3 separate bugs** to fix to enable correct LRLP distillation.

---

### Bug 1 — Asymmetric Input Routing (`DistillationModelLRLP`)

**File created:** `PaddleOCR/ppocr/modeling/architectures/distillation_model_lrlp.py`

**Root cause:** `DistillationModel.forward()` passes the single input tensor `x` to every sub-model, regardless of whether it is Teacher or Student:

```python
# Original DistillationModel.forward — BROKEN for LRLP
for idx, model_name in enumerate(self.model_name_list):
    result_dict[model_name] = self.model_list[idx](x, data)
    #                                               ↑
    #                           same tensor for Teacher and Student
```

**The fix:** `DistillationModelLRLP` subclasses `DistillationModel` and overrides `forward()` to route LR to Student and HR to Teacher. The HR tensor `x_hr` is passed as the last element of `KeepKeys`, so no changes to the data pipeline format are needed:

```python
# distillation_model_lrlp.py
class DistillationModelLRLP(DistillationModel):
    def forward(self, x, data=None):

        # --- Evaluation: only run the Student ---
        # Teacher adds no value at inference; skipping it halves eval time.
        if not self.training:
            student_out = self.model_list[self._student_idx](x, data)
            return {"Student": student_out, "Teacher": student_out}

        # --- Training: split HR from data ---
        x_lr         = x           # LR image for Student
        x_hr         = data[-1]    # HR image tensor, appended by KeepKeys as last element
        data_student = data[:-1]   # everything except image_hr (labels, length, etc.)

        result_dict = {}
        for idx, model_name in enumerate(self.model_name_list):
            model = self.model_list[idx]

            if model_name == "Teacher":
                # Teacher runs on HR and must not accumulate gradients
                was_training = model.training
                model.train()   # force BN layers into train mode for correct running stats
                with paddle.no_grad():
                    result_dict[model_name] = model(x_hr, data_student)
                if not was_training:
                    model.eval()
            else:
                # Student trains normally on LR
                result_dict[model_name] = model(x_lr, data_student)

        return result_dict
```

> **Why `model.train()` inside `no_grad()`?** BatchNorm behaves differently in train vs eval mode. Calling `model.eval()` would freeze running mean/var, making the Teacher's predictions inconsistent. We force train mode *only for BN statistics* while blocking gradient computation with `no_grad()`.

The config must include `image_hr` as the **last** key in `KeepKeys`:

```yaml
# distill config
Train:
  dataset:
    transforms:
      - DecodeMultiImagePair:   # decodes both LR and HR stacks
          img_mode: BGR
      - RecResizeImgPair:       # resizes both LR and HR to target sizes
      - KeepKeys:
          keep_keys: [image, label, length, image_hr]   # image_hr must be LAST
```

---

### Bug 2 — Broken NRTR DML Loss (`DistillationNRTRDMLLossLRLP`)

**File created:** `PaddleOCR/ppocr/losses/distillation_loss_lrlp.py`

**Root cause:** The standard `DistillationNRTRDMLLoss` has a subtle MRO (Method Resolution Order) bug. Its `forward` calls `super().forward(out1, out2, non_pad_mask)`. Due to Python's MRO in PaddleOCR's loss hierarchy, `super()` resolves to `DistillationDMLLoss.forward(predicts, batch)` — not `DMLLoss.forward(tensor1, tensor2)` as intended. The tensor `out1` is misinterpreted as a `predicts` dict, which causes:

```
ValueError: only one element tensors can be converted to Python scalars
```

**The fix:** Create a new subclass that calls `DMLLoss.forward` *explicitly* by name, bypassing the broken MRO:

```python
# distillation_loss_lrlp.py
from ppocr.losses.distillation_loss import DistillationDMLLoss
from ppocr.losses.rec_dml_loss import DMLLoss

class DistillationNRTRDMLLossLRLP(DistillationDMLLoss):
    def forward(self, predicts, batch):
        # Extract Teacher and Student output logits
        out1 = predicts.get(self.model_name_pairs[0][0])
        out2 = predicts.get(self.model_name_pairs[0][1])

        if self.multi_head:
            # For multi-head models, pick the correct head output
            loss = DMLLoss.forward(self, out1[self.dis_head], out2[self.dis_head])
        else:
            loss = DMLLoss.forward(self, out1, out2)

        return {"loss_dml_nrtr": loss}
```

Note that `non_pad_mask` — present in the original class — is removed entirely. It is not needed for DML (soft distribution matching), which operates on raw logit tensors, not attention masks.

---

### Bug 3 — Paired LR/HR Data Loading (`DistillationDataSet`)

**File created:** `PaddleOCR/ppocr/data/distillation_dataset.py`

**Root cause:** All existing PaddleOCR dataset classes load a single image per label line. Distillation requires **two image sequences per line**: an LR sequence (for Student input) and an HR sequence (for Teacher input), which may have different spatial dimensions.

**Label file format:**

```
lr_f1.png,lr_f2.png,...,lr_f5.png|hr_f1.png,hr_f2.png,...,hr_f5.png\tplate_label
```

The `|` separates LR paths from HR paths; `,` separates frames within each group.

**The fix:** `DistillationDataSet` subclasses `MultiScaleDataSet` and overrides loading and resizing:

```python
# distillation_dataset.py
class DistillationDataSet(MultiScaleDataSet):

    def _parse_lr_hr(self, file_name):
        """'lr1,lr2,...|hr1,hr2,...' → {"lr": [...], "hr": [...]}"""
        lr_s, hr_s = file_name.split("|", 1)
        return {"lr": lr_s.split(","), "hr": hr_s.split(",")}

    def _load_images_from_paths(self, paths):
        """Read a list of image paths as bytes."""
        result = []
        for p in paths:
            full = os.path.join(self.data_dir, p)
            if not os.path.exists(full):
                return None
            with open(full, "rb") as f:
                result.append(f.read())
        return result

    def _resize_both(self, data, imgW, imgH):
        """
        Resize LR with aspect-ratio padding (preserves natural plate proportions).
        Resize HR to exact (imgW, imgH) without padding (Teacher expects fixed size).
        """
        # LR: standard resizing with width padding
        data = super().resize_norm_img(data, imgW, imgH, padding=True)

        # HR: force exact resize, no padding
        if "image_hr" in data and data["image_hr"] is not None:
            hr_imgs = data["image_hr"]
            resized = [cv2.resize(f, (imgW, imgH)) for f in hr_imgs]
            hr_stack = np.stack(resized, axis=0).transpose((0, 3, 1, 2)) / 255.0
            hr_stack = (hr_stack - 0.5) / 0.5
            data["image_hr"] = hr_stack.astype("float32")
        return data
```

Key behaviors:

| Mode | Behavior |
|:---|:---|
| **Train** | Loads both LR and HR, calls `_resize_both`, returns `image` (LR) + `image_hr` (HR) |
| **Eval** | Loads only LR (Student path); if HR is present, resizes it to match LR dimensions |
| **Retry** | Both modes retry up to `MAX_RETRY = 5` times on any loading or decoding failure |

---

### Registration — Connecting the Pieces

All three custom classes must be imported in the framework's registries. These imports have already been added:

| Registry File | Import Added |
|:---|:---|
| `ppocr/modeling/architectures/__init__.py` | `from .distillation_model_lrlp import DistillationModelLRLP` |
| `ppocr/losses/combined_loss.py` | `from .distillation_loss_lrlp import DistillationNRTRDMLLossLRLP` |
| `ppocr/data/__init__.py` | `from ppocr.data.distillation_dataset import DistillationDataSet` |
| `ppocr/data/imaug/__init__.py` | `from .distillation_ops import DecodeMultiImagePair, RecResizeImgPair` |

### Config Diff — Standard vs LRLP

| Config Field | Standard PaddleOCR | LRLP Custom |
|:---|:---|:---|
| `Architecture.name` | `DistillationModel` | `DistillationModelLRLP` |
| `Loss: NRTR DML` | `DistillationNRTRDMLLoss` | `DistillationNRTRDMLLossLRLP` |
| `Train.dataset.name` | `MultiScaleDataSet` | `DistillationDataSet` |
| `Train.transforms[0]` | `DecodeImage` | `DecodeMultiImagePair` |
| `KeepKeys` | no `image_hr` | `image_hr` **must be last** |

---

