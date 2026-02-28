#!/bin/bash
set -e

FILE_ID="1_1CR-ZUgs1BZFF00e9KGqEEk4isGgCUr"
OUTPUT_ZIP="processed.zip"
TARGET_DIR="data"
echo "Downloading processed dataset from Google Drive..."
gdown --fuzzy "https://drive.google.com/file/d/${FILE_ID}/view?usp=drive_link" -O "${OUTPUT_ZIP}"

echo "Extracting to ${TARGET_DIR}..."
unzip -o -q "${OUTPUT_ZIP}" -d "${TARGET_DIR}"

echo "Cleaning up..."
rm "${OUTPUT_ZIP}"

echo "✓ Done! Dataset extracted to ${TARGET_DIR}/preprocessed"