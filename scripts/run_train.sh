#!/bin/bash
# YOLOv8 Model Training Script

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit

echo "========================================================"
echo "Current Directory: $PWD"
echo "========================================================"
echo ""

read -p "Run preprocessing first? (Y/N, default is N): " run_pre
if [[ "$run_pre" =~ ^[Yy]$ ]]; then
    echo ""
    echo "========================================================"
    echo "Running Preprocessing..."
    echo "========================================================"
    python src/preprocess.py --config config.yaml
    if [ $? -ne 0 ]; then
        echo "[Error] Preprocessing failed."
        exit 1
    fi
fi

echo ""
echo "========================================================"
echo "Current Configuration Parameters:"
echo "========================================================"
python scripts/show_config.py
echo ""

# --- Guard: check preprocessed data exists ---
# NOTE: This path must match the 'processed' path resolved by config.yaml + get_paths()
if [ ! -d "preprocessed_data/images/train" ]; then
    echo "[Warning] preprocessed_data/images/train not found."
    echo "          Consider running preprocessing first (select Y at the next prompt)."
    echo ""
fi

read -p "Proceed with training using these parameters? (Y/N, default is N): " run_train
if [[ ! "$run_train" =~ ^[Yy]$ ]]; then
    echo "Training cancelled by user."
    exit 0
fi

echo ""

TRAIN_CMD="python src/train.py --config config.yaml"
if [ -f "runs/train/weights/last.pt" ]; then
    echo "========================================================"
    echo "[Notice] Found a previous training checkpoint (last.pt)."
    echo "========================================================"
    read -p "Do you want to resume training from this checkpoint? (Y/N, default is N): " do_resume
    if [[ "$do_resume" =~ ^[Yy]$ ]]; then
        TRAIN_CMD="python src/train.py --config config.yaml --resume"
    fi
    echo ""
fi

echo "========================================================"
echo "Starting YOLOv8 Model Training..."
echo "========================================================"
$TRAIN_CMD
if [ $? -ne 0 ]; then
    echo ""
    echo "[Error] Training encountered an error."
    exit 1
fi

echo ""
echo "========================================================"
echo "Training Execution Finished."
echo "========================================================"
