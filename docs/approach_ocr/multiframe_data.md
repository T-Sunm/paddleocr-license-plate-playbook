# Step 3 — Multi-Frame Data Pipeline

Two changes are needed so the data loader can feed 5-frame stacks to the model.

#### `DecodeMultiImage` — Stack Multiple Frames

**File modified:** `PaddleOCR/ppocr/data/imaug/operators.py`

The standard `DecodeImage` decodes a single `bytes` object. `DecodeMultiImage` handles a `list[bytes]` — one element per frame — and assembles them into a stacked tensor `[T, H, W, C]`:

```python
class DecodeMultiImage(object):
    """
    Decodes a list of image byte buffers, pads all frames to the same (H, W),
    and stacks them into [T, H, W, C].

    data["image"] must be list[bytes] — one bytes object per frame.
    """
    def __init__(self, img_mode="RGB", channel_first=False, pad_value=0, **kwargs):
        ...

    def __call__(self, data):
        img_list = data["image"]  # list[bytes]

        decoded_imgs = []
        for img_bytes in img_list:
            img_np = np.frombuffer(img_bytes, dtype="uint8")
            img    = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None:
                return None
            if self.img_mode == "RGB":
                img = img[:, :, ::-1]
            decoded_imgs.append(img)

        # Pad all frames to the maximum (H, W) across the batch
        H = max(im.shape[0] for im in decoded_imgs)
        W = max(im.shape[1] for im in decoded_imgs)
        padded = []
        for im in decoded_imgs:
            h, w = im.shape[:2]
            if h < H or w < W:
                im = cv2.copyMakeBorder(im, 0, H-h, 0, W-w,
                                        cv2.BORDER_CONSTANT, value=self.pad_value)
            padded.append(im)

        data["image"] = np.stack(padded, axis=0)  # [T, H, W, C]
        return data
```

#### `NormalizeImage` — 4D Broadcasting Fix

**File modified:** `PaddleOCR/ppocr/data/imaug/operators.py`

The existing `NormalizeImage` broadcasts `mean` and `std` assuming a 3D image `[H, W, C]` (or `[C, H, W]`). For 4D stacks `[T, C, H, W]`, the shapes must be explicitly extended:

```python
def __call__(self, data):
    img = data["image"]
    is_4d = len(img.shape) == 4

    if is_4d:
        # Reshape mean/std to broadcast across T frames
        if self.mean.shape == (3, 1, 1):         # CHW order
            mean = self.mean.reshape((1, 3, 1, 1))   # → [1, C, 1, 1]
            std  = self.std.reshape((1, 3, 1, 1))
        else:                                     # HWC order
            mean = self.mean.reshape((1, 1, 1, 3))   # → [1, 1, 1, C]
            std  = self.std.reshape((1, 1, 1, 3))
        data["image"] = (img.astype("float32") * self.scale - mean) / std
    else:
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std

    return data
```

#### `SimpleDataSet` — Multi-Path Filename Parsing

**File modified:** `PaddleOCR/ppocr/data/simple_dataset.py`

The label file for multiframe uses comma-separated paths in the filename column. Two methods handle this:

**`_try_parse_filename_list`** — parses the filename column into one of three types:

```python
def _try_parse_filename_list(self, file_name):
    """
    Supports three label file formats:

    1. Single image:  "img.png"
       Returns: "img.png" (str)

    2. Multiframe:    "f1.png,f2.png,f3.png,f4.png,f5.png"
       Returns: ["f1.png", "f2.png", "f3.png", "f4.png", "f5.png"] (list)

    3. Distill pair:  "lr1,lr2,...|hr1,hr2,..."
       Returns: {"lr": [...], "hr": [...]} (dict)
    """
    if len(file_name) > 0:
        if file_name[0] in ("[", "{"):         # JSON format
            try:
                return json.loads(file_name)
            except:
                pass
        elif "|" in file_name:                 # Distill pair format
            parts = file_name.split("|")
            return {"lr": parts[0].split(","), "hr": parts[1].split(",")}
        elif "," in file_name:                 # Multiframe format
            return file_name.split(",")
    return file_name
```

**`_load_image`** — handles all three parsed types safely:

```python
def _load_image(self, file_name_info):
    if isinstance(file_name_info, (list, dict)):
        # Determine which paths to load
        if isinstance(file_name_info, dict):
            # Dict routing: prefer 'hr' keys if present, else 'lr'
            paths = file_name_info.get("hr", file_name_info.get("lr", []))
        else:
            paths = file_name_info

        img_list = []
        for p in paths:
            img_path = os.path.join(self.data_dir, p)
            if not os.path.exists(img_path):
                return None, None   # fail safely, triggers retry in __getitem__
            with open(img_path, "rb") as f:
                img_list.append(f.read())
        return img_list, file_name_info
    else:
        # Standard single-image path
        img_path = os.path.join(self.data_dir, file_name_info)
        if not os.path.exists(img_path):
            return None, None
        with open(img_path, "rb") as f:
            return f.read(), img_path
```

**`MultiScaleDataSet.resize_norm_img`** — resizes 4D stacks correctly:

```python
def resize_norm_img(self, data, imgW, imgH, padding=True):
    img   = data["image"]
    is_4d = len(img.shape) == 4   # [T, H, W, C]

    if is_4d:
        T, h, w, c = img.shape
        # Resize each frame independently
        resized_imgs = [
            cv2.resize(img[t], (resized_w, imgH)) for t in range(T)
        ]
        resized_image = np.stack(resized_imgs, axis=0)          # [T, H', W, C]
        resized_image = resized_image.transpose((0, 3, 1, 2)) / 255  # [T, C, H', W]
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((T, 3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :, :resized_w] = resized_image         # pad width

    data["image"] = padding_im
    return data
```

### Usage in Config (Multiframe)

```yaml
# configs/rec_v5_multiframe.yml

Backbone:
  name: PPHGNetV2_B4
  use_temporal: true           # ← activates all temporal branches
  frozen_stages: 1             # ← freeze stem + stage0 + stage1; train temporal parts

Train:
  dataset:
    name: SimpleDataSet
    label_file_list:
      - data/multiframe/train_multiframe.txt  # "f1,f2,f3,f4,f5\tlabel"
    transforms:
      - DecodeMultiImage:                     # ← replaces DecodeImage
          img_mode: BGR
          channel_first: false
      - NormalizeImage:
          scale: '1./255.'
          mean: [0.5, 0.5, 0.5]
          std:  [0.5, 0.5, 0.5]
          order: chw
```

---

