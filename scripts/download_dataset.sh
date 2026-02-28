#!/bin/bash

# ============================================
# Script to download ICPR License Plate Dataset
# ============================================

set -e  # Exit on error


DATA_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DATA_DIR"

echo "Starting dataset download..."

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

# Download dataset
echo "Downloading dataset..."
cd "$DATA_DIR"
gdown 1pD-8BDR1izKBrU5y2krWIXe5dmUvug28

# Extract dataset
echo "Extracting dataset..."
unzip -q wYe7pBJ7-train.zip

# Clean up
echo "Cleaning up..."
rm -f wYe7pBJ7-train.zip

echo "Dataset download complete."
echo "Location: $DATA_DIR/wYe7pBJ7-train"

