#!/bin/bash
# =============================================================================
# PaddleOCR Approach - Environment Setup
# =============================================================================
# This script ONLY sets up the Python environment for PaddleOCR.
# For full project setup (data, pretrained models, etc.), use:
#   bash scripts/setup.sh
# =============================================================================

set -e

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PaddleOCR Approach - Environment Setup                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "[1/2] Creating virtual environment..."
    uv venv --python 3.10
else
    echo "[1/2] Virtual environment already exists."
fi

# Sync dependencies
echo "[2/2] Installing dependencies..."
uv sync

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ PaddleOCR environment ready!                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Use uv run (no manual activation needed):"
echo "  uv run python scripts/..."
echo ""
echo "For full project setup (data, pretrained models, configs):"
echo "  bash scripts/setup.sh"
