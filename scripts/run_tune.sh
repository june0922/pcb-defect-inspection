#!/bin/bash
# YOLOv8 Hyperparameter Tuning Script

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit 1

echo "========================================================"
echo " YOLOv8 Hyperparameter Tuning"
echo " Current Directory: $PWD"
echo "========================================================"
echo ""

# --- Preprocessing step (optional) ---
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
    echo "[Error] preprocessed_data/images/train not found."
    echo "        Please run preprocessing first."
    exit 1
fi

# --- Guard: check model weights exist ---
if [ ! -f "weights/yolov8n.pt" ]; then
    echo "[Warning] weights/yolov8n.pt not found."
    echo "          Make sure the base model weight file exists before tuning."
    echo ""
fi

# --- Confirm tuning ---
read -p "Proceed with hyperparameter tuning? (Y/N, default is N): " run_tune
if [[ ! "$run_tune" =~ ^[Yy]$ ]]; then
    echo "Tuning cancelled by user."
    exit 0
fi

echo ""

# --- Guard: check previous tune results ---
if [ -d "runs/tune" ]; then
    echo "========================================================"
    echo "[Notice] Previous tuning results found in runs/tune/."
    echo "  O = Overwrite (start tuning from scratch)"
    echo "  N = Cancel (keep existing results)"
    echo "========================================================"
    read -p "Your choice (O/N, default is N): " tune_action

    if [[ "$tune_action" =~ ^[Oo]$ ]]; then
        echo "Overwriting previous tuning results..."
    else
        echo "Tuning cancelled by user."
        exit 0
    fi
    echo ""
fi

echo "========================================================"
echo "Starting YOLOv8 Hyperparameter Tuning..."
echo "[Note] This process runs multiple short training cycles."
echo "       Estimated time: iterations x epochs duration."
echo "========================================================"
python src/tune.py --config config.yaml
if [ $? -ne 0 ]; then
    echo ""
    echo "[Error] Hyperparameter tuning encountered an error."
    exit 1
fi

echo ""
echo "========================================================"
echo "Hyperparameter Tuning Finished."
echo "Results saved to: runs/tune/"
echo "Best hyperparameters: runs/tune/best_hyperparameters.yaml"
echo "========================================================"
