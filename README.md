# Plate Recognition — Low Resolution

A license plate recognition system for low-resolution videos, consisting of two sequential processing stages:

```
LR frames  →  Super Resolution (SR)  →  OCR Recognition  →  Plate text
```

Built upon **PaddleOCR**, featuring customized architectures for the low-resolution license plate (LRLP) problem.

---

## Module Documentation

| Module | Documentation | Content |
|:---|:---|:---|
| **Super Resolution** | [`docs/approach_sr/README.md`](./docs/approach_sr/README.md) | LMDB build, Gestalt/Telescope training, STN fix for 48×320, weight prefix fix |
| **OCR Recognition** | [`docs/approach_ocr/README.md`](./docs/approach_ocr/README.md) | Ensemble, Multiframe, Distillation, temporal architecture, custom distillation classes |

---

## Directory Structure

```
plate-recognition-low-resolution/
│
├── PaddleOCR/                    # PaddleOCR Submodule (patched)
│
├── configs/
│   ├── sr/                       # Configurations for SR (Gestalt, Telescope)
│   ├── ensemble/                 # 5-fold ensemble configs (det + rec)
│   ├── multiframe/               # Multiframe SFT config
│   ├── teacher/                  # Teacher model config (distillation)
│   └── distill/                  # Student distillation config
│
├── scripts/
│   ├── data/                     # Data processing pipeline
│   │   ├── crop_plates.py        # Crop images based on bounding boxes
│   │   ├── augment_data.py       # Augmentation to simulate LR
│   │   ├── build_dataset.py      # Create train/val/test splits
│   │   ├── build_lmdb_sr.py      # Build LMDB for SR
│   │   ├── build_multiframe_dataset.py
│   │   ├── build_distill_dataset.py
│   │   └── prepare_ensemble.py   # Split 5-fold ensemble
│   │
│   ├── train/
│   │   └── train_sr.py           # Train SR (Gestalt / Telescope)
│   │
│   ├── infer/
│   │   ├── infer_sr.py           # Inference SR (checkpoint mode)
│   │   ├── infer_single.py       # Inference OCR single-frame
│   │   └── infer_multiframe.py   # Inference OCR multiframe (5 frames)
│   │
│   ├── tools/
│   │   ├── generate_configs.py         # Generate ensemble configs
│   │   ├── generate_distill_configs.py
│   │   ├── generate_multiframe_configs.py
│   │   ├── download_pretrained.py
│   │   ├── fix_sr_pretrained.py        # Fix prefix weight for SR
│   │   ├── export_ocr_models.py
│   │   ├── evaluate_ocr_models.py
│   │   └── verify_freeze.py
│   │
│   ├── setup.sh                  # Full pipeline setup (env + data + configs)
│   └── run.sh                    # Wrapper to run scripts from project root
│
├── src/
│   └── plate_ocr/               # Internal Python package (paths, runner, utils)
│
├── data/
│   ├── raw/                      # Raw dataset (not committed)
│   ├── preprocessed/             # Preprocessed images + bboxes
│   ├── preprocessed_cropped/     # Cropped images based on bboxes
│   ├── processed_rec/            # OCR dataset (train/val/test .txt)
│   ├── processed_rec_multiframe/ # Multiframe dataset
│   ├── processed_rec_distill/    # Distillation dataset (LR|HR pairs)
│   ├── lmdb_sr/                  # LMDB dataset for SR
│   └── english_decomposition.txt # Dictionary for StrokeFocusLoss
│
├── output/                       # Checkpoints & logs (not committed)
├── weights/                      # Pretrained weights with fixed prefixes
├── docs/
│   ├── approach_sr/               # SR documentation
│   │   ├── README.md
│   │   ├── input_alignment.md
│   │   ├── stn_fix.md
│   │   ├── selective_freeze.md
│   │   └── prefix_remapping.md
│   └── approach_ocr/              # OCR documentation
│       ├── README.md
│       ├── backbone_freeze.md
│       ├── head_channels.md
│       ├── multiframe_arch.md
│       ├── multiframe_data.md
│       └── distillation_lrlp.md
├── setup.sh                      # PaddleOCR environment setup (.venv)
└── pyproject.toml
```

---

## Image Processing Flow

```
Input: Video / track directory (5 LR frames)
          │
          ▼
  ┌────────────────────────┐
  │  Super Resolution      │  (Gestalt TSRN, 48×320)
  │  scripts/infer/        │  LR → Enhanced HR
  │  infer_sr.py           │
  └────────┬───────────────┘
           │ HR frames (48×320)
           ▼
  ┌─────────────────────────────────┐
  │  OCR Recognition                │
  │  • Single-frame: infer_single   │  → text + confidence
  │  • Multiframe:  infer_multiframe│  → text + confidence
  └─────────────────────────────────┘
           │
           ▼
  Plate text (e.g. "51A-12345")
```

**Two OCR inference modes:**

- **Single-frame** (`infer_single.py`): Runs the 5-model ensemble, employing majority voting on individual frames.
- **Multiframe** (`infer_multiframe.py`): Receives 5 frames simultaneously, leveraging temporal redundancy through a dual-branch 3D/2D architecture.

---

## 1. Environment Setup

### 1.1 Main Environment (PaddleOCR — training & inference)

```bash
# Create Python 3.10 venv + install dependencies
bash setup.sh
```

Then use `uv run python ...` to run the scripts (no manual activation needed).

### 1.2 Preprocessor Environment (Qwen3-VL + Flash Attention)

Used for heavy data pipeline scripts (`preprocess_resize.py`, notebooks):

```bash
# Create separate venv
uv venv .venv-preprocess --python 3.10
source .venv-preprocess/bin/activate

# PyTorch + CUDA 12.4
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# Flash Attention 2 (pre-built wheel for torch 2.6)
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Remaining libraries
uv pip install transformers==4.57.5 accelerate==1.12.0 \
  pillow==12.0.0 opencv-python==4.12.0.88 albumentations==2.0.8 \
  supervision==0.27.0 ipykernel matplotlib==3.10.8
```

---

## 2. Data Pipeline

### 2.1 Full automated setup (recommended)

```bash
bash scripts/setup.sh
```

This script automatically executes all 7 steps: Environment → Download → Crop → Augment → Build Dataset → Ensemble Prep → Generate Configs.

### 2.2 Manual step-by-step

```bash
# Step 1: Download preprocessed data
bash scripts/download_processed.sh

# Step 2: Download PP-OCRv5 pretrained weights
uv run python scripts/tools/download_pretrained.py --mode both

# Step 3: Crop images based on bounding boxes
uv run python scripts/data/crop_plates.py \
  --input data/preprocessed --output data/preprocessed_cropped

# Step 4: Augmentation to simulate LR
uv run python scripts/data/augment_data.py \
  --input data/preprocessed_cropped

# Step 5: Build datasets (train/val/test splits)
uv run python scripts/data/build_dataset.py --mode all

# Step 6: Prepare ensemble 5-fold data
uv run python scripts/data/prepare_ensemble.py --mode all

# Step 7: Generate config files
uv run python scripts/tools/generate_configs.py --mode det
uv run python scripts/tools/generate_configs.py --mode rec
```

---

## 3. Super Resolution

> Full details: **[docs/approach_sr/README.md](./docs/approach_sr/README.md)**

### Train

```bash
# Build LMDB dataset (run from project root, use .venv-preprocess)
.venv-preprocess/bin/python scripts/data/build_lmdb_sr.py

# Fix prefix of downloaded weights
uv run python scripts/tools/fix_sr_pretrained.py \
  --input weights/sr_pretrained/best_accuracy.pdparams \
  --output weights/sr_pretrained/best_accuracy_fixed.pdparams \
  --algo gestalt

# Train Gestalt (recommended)
uv run python scripts/train/train_sr.py --algo gestalt

# Train Telescope
uv run python scripts/train/train_sr.py --algo telescope
```

### Inference

```bash
uv run python scripts/infer/infer_sr.py \
  --image_dir data/sample_plates/ \
  --checkpoint output/sr_gestalt_XXXXXX/best_accuracy \
  --output_dir output/inference/sr_result
```

---

## 4. OCR Recognition

> Full details: **[docs/approach_ocr/README.md](./docs/approach_ocr/README.md)**

### Train Ensemble (5-fold)

```bash
uv run python scripts/tools/generate_configs.py --mode rec
# Or use the wrapper:
./scripts/run.sh train_ensemble.py --mode rec
```

### Train Multiframe SFT

```bash
# Build multiframe dataset (use .venv-preprocess)
.venv-preprocess/bin/python scripts/data/build_multiframe_dataset.py

uv run python scripts/tools/generate_multiframe_configs.py
./scripts/run.sh train_multiframe.py
```

### Train Distillation

```bash
# Step 1: Build distillation dataset
uv run python scripts/data/build_distill_dataset.py

# Step 2: Train teacher
uv run python scripts/tools/generate_distill_configs.py --mode teacher
./scripts/run.sh train_distill.py --mode teacher

# Step 3: Train student with distillation
uv run python scripts/tools/generate_distill_configs.py \
  --mode distill --teacher-path output/teacher_v5_XXXXXX/best_accuracy
./scripts/run.sh train_distill.py --mode distill
```

### Inference

```bash
# Single-frame (ensemble)
uv run python scripts/infer/infer_single.py \
  --gt data/processed_rec/test_rec.txt

# Multiframe (5 frames from 1 track)
uv run python scripts/infer/infer_multiframe.py \
  --track_dir data/preprocessed_cropped/track_001 \
  --config configs/multiframe/rec_v5_multiframe.yml \
  --checkpoint output/multiframe_v5_XXXXXX/best_accuracy

# Batch inference with a label file
uv run python scripts/infer/infer_multiframe.py \
  --label_file data/processed_rec_multiframe/test_multiframe.txt \
  --config configs/multiframe/rec_v5_multiframe.yml \
  --checkpoint output/multiframe_v5_XXXXXX/best_accuracy \
  --output output/eval_result.csv \
  --zip output/submission.zip
```

---

## 5. Backup & Restore

It's only necessary to back up two critical directories:

```bash
# Backup
tar -czvf project_backup.tar.gz data/raw output/

# The rest can be reproduced:
#   data/processed_*/          ← re-run scripts/data/
#   weights/ pretrain_models/  ← re-run scripts/tools/download_pretrained.py
#   configs/ensemble/          ← re-run scripts/tools/generate_configs.py
```