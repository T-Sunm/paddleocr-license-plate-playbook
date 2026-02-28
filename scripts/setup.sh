#!/bin/bash
# =============================================================================
# PLATE RECOGNITION - FULL DATA PIPELINE SETUP
# =============================================================================
# This script orchestrates the entire data preparation pipeline:
# Environment -> Download -> Crop -> Augment -> Build Dataset -> Ensemble
# =============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PADDLE_DIR="$PROJECT_ROOT/approaches/paddle"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Plate Recognition - Full Automated Data Pipeline       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 1. Environment Setup
echo -e "${YELLOW}[Step 1/7] Setting up environment...${NC}"
cd "$PADDLE_DIR"
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment and installing dependencies..."
    uv venv --python 3.10
    uv sync
fi
source .venv/bin/activate
echo -e "${GREEN}  ✓ Environment ready${NC}"

# 2. Download Preprocessed Data (Scenario A & B with BBoxes)
echo -e "\n${YELLOW}[Step 2/7] Checking preprocessed data...${NC}"
if [ ! -d "$PROJECT_ROOT/data/preprocessed" ]; then
    echo "  Data not found. Downloading preprocessed dataset (Scenario A & B)..."
    bash "$PROJECT_ROOT/scripts/download_processed.sh"
else
    echo -e "${GREEN}  ✓ Preprocessed data already exists in data/preprocessed${NC}"
fi

# 3. Pretrained Models
echo -e "\n${YELLOW}[Step 3/7] Checking pretrained models...${NC}"
if [ ! -d "$PROJECT_ROOT/pretrain_models/paddleocr_v5_det" ]; then
    echo "  Downloading PP-OCRv5 pretrained models..."
    uv run python scripts/download_pretrained.py --mode both
else
    echo -e "${GREEN}  ✓ Pretrained models exist${NC}"
fi

# 4. Crop Images (Bounding Box)
echo -e "\n${YELLOW}[Step 4/7] Cropping plates from bounding boxes...${NC}"
uv run python "$PROJECT_ROOT/scripts/crop_plates.py" \
    --input "$PROJECT_ROOT/data/preprocessed" \
    --output "$PROJECT_ROOT/data/preprocessed_cropped"
echo -e "${GREEN}  ✓ Cropping complete${NC}"

# 5. Data Augmentation
echo -e "\n${YELLOW}[Step 5/7] Running data augmentation on cropped images...${NC}"
# This applies LRLPR effects to simulate low resolution
uv run python "$PROJECT_ROOT/scripts/augment_data.py" \
    --input "$PROJECT_ROOT/data/preprocessed_cropped"
echo -e "${GREEN}  ✓ Augmentation complete${NC}"

# 6. Build Base Dataset (Train/Val/Test Split)
echo -e "\n${YELLOW}[Step 6/7] Building unified datasets...${NC}"
uv run python "$PROJECT_ROOT/scripts/build_dataset.py" --mode all
echo -e "${GREEN}  ✓ Base datasets built (data/processed_det, data/processed_rec)${NC}"

# 7. Prepare Ensemble Splits & Configs
echo -e "\n${YELLOW}[Step 7/7] Preparing ensemble data and configurations...${NC}"
uv run python "$PROJECT_ROOT/scripts/prepare_ensemble.py" --mode all

echo "  Generating training configurations for ensemble..."
uv run python scripts/generate_configs.py --mode det
uv run python scripts/generate_configs.py --mode rec
echo -e "${GREEN}  ✓ Ensemble data and configs ready${NC}"

echo -e "\n${BLUE}================================================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETE! SYSTEM READY FOR TRAINING.${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Next steps:"
echo -e "  1. Train Detection Ensemble:   ${YELLOW}./scripts/run.sh train_ensemble.py --mode det${NC}"
echo -e "  2. Train Recognition Ensemble: ${YELLOW}./scripts/run.sh train_ensemble.py --mode rec${NC}"
echo ""
echo "Check progress in: approaches/paddle/output/ensemble_v5"
