# PaddleOCR Approach - License Plate Recognition

This document covers everything you need to train, tune, and extend the PaddleOCR-based recognition pipeline for license plate recognition on low-resolution video. It supports three training paradigms:

| Paradigm | Description |
|:---|:---|
| **Ensemble (5-fold)** | Five independent single-frame models trained on different data splits, predictions merged at inference. |
| **Multiframe SFT** | One model that sees **5 consecutive frames** of the same plate, leveraging temporal redundancy. |
| **Distillation (LRLP)** | A Teacher (trained on HR) supervises a Student (trains on LR). Bridges the SR-OCR gap. |

> **If you re-clone `PaddleOCR/`** as a fresh submodule, all source-level patches described here must be re-applied. Every modification is intentional — this document explains the *why* behind each change.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Scripts Reference](#scripts-reference)
3. [PaddleOCR Modifications Overview](#paddleocr-modifications-overview)
4. [Config Reference](#config-reference)

## Quick Start

```bash
# --- Ensemble (5-fold) ---
uv run python scripts/tools/generate_configs.py --mode rec
./scripts/run.sh train_ensemble.py --mode rec

# --- Multiframe SFT ---
# Step 1: Build multiframe dataset
.venv-preprocess/bin/python scripts/data/build_multiframe_dataset.py
# Step 2: Generate config & train
uv run python scripts/tools/generate_multiframe_configs.py
./scripts/run.sh train_multiframe.py

# --- Distillation (LRLP) ---
uv run python scripts/tools/generate_distill_configs.py --mode teacher
./scripts/run.sh train_distill.py --mode teacher
uv run python scripts/tools/generate_distill_configs.py --mode distill \
    --teacher-path output/teacher_v5_XXXXXX/best_accuracy
./scripts/run.sh train_distill.py --mode distill

# --- Inference ---
uv run python scripts/infer/infer_single.py --gt data/processed_rec/test_rec.txt
```

---

## Scripts Reference

### Training Pipeline

| Script | Location | Purpose |
|:---|:---|:---|
| `generate_configs.py` | `scripts/tools/` | Generate 5-fold ensemble YAML configs from template |
| `generate_distill_configs.py` | `scripts/tools/` | Generate teacher or distillation config |
| `generate_multiframe_configs.py` | `scripts/tools/` | Generate multiframe SFT config (5-frame input) |
| `train_ensemble.py` | via `run.sh` | Train detection/recognition ensemble models |
| `train_distill.py` | via `run.sh` | Train teacher or distillation student model |
| `train_multiframe.py` | via `run.sh` | Train multiframe recognition model |

### Inference & Evaluation

| Script | Location | Purpose |
|:---|:---|:---|
| `infer_single.py` | `scripts/infer/` | Single-frame inference on a label file |
| `infer_multiframe.py` | `scripts/infer/` | Multiframe inference (single track or batch) |
| `evaluate_ocr_models.py` | `scripts/tools/` | Batch evaluation via PaddleOCR `tools/eval.py` |

### Utilities

| Script | Location | Purpose |
|:---|:---|:---|
| `utils.py` | `src/plate_ocr/` | NED metric, checkpoint discovery, output parsing |
| `download_pretrained.py` | `scripts/tools/` | Download PP-OCRv5 Server pretrained weights |
| `verify_freeze.py` | `scripts/tools/` | Debug tool to verify frozen backbone parameter counts |

---

## PaddleOCR Modifications Overview

### Feature 1 — Transfer Learning via Backbone Stage Freezing
> Adds `frozen_stages` to PPHGNetV2 and LCNetV3 to prevent overfitting on small plate datasets.
→ [Full documentation](backbone_freeze.md)

### Feature 2 — Dynamic Recognition Head Channels
> Fixes hardcoded `out_channels=97` in MultiHead to support custom character sets (38 chars for plates).
→ [Full documentation](head_channels.md)

### Feature 3 — Multiframe Temporal Fusion Architecture
> Extends PPHGNetV2 to process 5 consecutive frames using a hybrid spatial-temporal dual-branch design.
- [Architecture & Modules](multiframe_arch.md) — building blocks, wiring into backbone
- [Data Pipeline](multiframe_data.md) — dataset loading, decoding, normalization

### Feature 4 — Distillation on Asymmetric LR/HR Inputs (LRLP)
> Fixes 3 bugs in PaddleOCR's distillation pipeline to support Teacher(HR) → Student(LR) training.
→ [Full documentation](distillation_lrlp.md)

## Config Reference

### Template Files

| Template | Purpose |
|:---|:---|
| `configs/rec_v5_template.yaml` | Ensemble recognition (single-frame, 5-fold) |
| `configs/rec_v5_multiframe.yml` | Multiframe SFT (5-frame, `use_temporal: true`) |
| `configs/rec_v5_distill_template.yaml` | Distillation (teacher → student) |
| `configs/det_v5_template.yaml` | Ensemble detection |

### Generated Config Directories

After running `generate_configs.py`, configs land here:

| Directory | Contents |
|:---|:---|
| `configs/ensemble_v5/rec/` | `rec_v5_model_1.yml` … `rec_v5_model_5.yml` |
| `configs/multiframe_v5/` | `rec_v5_multiframe.yml` |
| `configs/teacher_v5/` | `rec_v5_teacher.yml` |
| `configs/distill_v5/` | `rec_v5_distill.yml` |
