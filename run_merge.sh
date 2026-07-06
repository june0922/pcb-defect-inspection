#!/bin/bash
cd "$(dirname "$0")"
echo "========================================"
echo "Starting Image Merge..."
echo "========================================"
python3 src/merge_images.py
if [ $? -ne 0 ]; then
    echo "[ERROR] An error occurred while running the script."
    exit 1
fi
echo "========================================"
echo "Process completed successfully."
exit 0
