#!/bin/bash
# Data Preprocessing Script

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment if it exists
if [ -f "$(dirname "$0")/../venv/bin/activate" ]; then
    source "$(dirname "$0")/../venv/bin/activate"
fi

echo "========================================================"
echo "Current Directory: $PWD"
echo "========================================================"
echo ""

echo "========================================================"
echo "Current Configuration Parameters:"
echo "========================================================"
python scripts/show_config.py
echo ""

read -p "Proceed with preprocessing? (Y/N, default is N): " run_pre
if [[ ! "$run_pre" =~ ^[Yy]$ ]]; then
    echo "Preprocessing cancelled by user."
    exit 0
fi

echo ""
echo "========================================================"
echo "Running Preprocessing..."
echo "========================================================"
python src/preprocess.py --config config.yaml
if [ $? -ne 0 ]; then
    echo ""
    echo "[Error] Preprocessing failed."
    exit 1
fi

echo ""
read -p "Proceed with Image Merge? (Y/N, default is N): " run_merge
if [[ ! "$run_merge" =~ ^[Yy]$ ]]; then
    echo "Image Merge cancelled by user."
else
    echo ""
    echo "========================================================"
    echo "Running Image Merge..."
    echo "========================================================"
    python src/merge_images.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "[Error] Image Merge failed."
        exit 1
    fi
fi

echo ""
echo "========================================================"
echo "Preprocessing Execution Finished."
echo "========================================================"
exit 0
