#!/bin/bash
# YOLOv8 K-Fold Cross Validation Script

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit 1

echo "========================================================"
echo " YOLOv8 K-Fold Cross Validation"
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
if [ ! -d "preprocessed_data/images/train" ]; then
    echo "[Error] preprocessed_data/images/train not found."
    echo "        Please run preprocessing first."
    exit 1
fi

# --- Confirm K-Fold training ---
read -p "Proceed with K-Fold cross validation? (Y/N, default is N): " run_kfold
if [[ ! "$run_kfold" =~ ^[Yy]$ ]]; then
    echo "K-Fold training cancelled by user."
    exit 0
fi

echo ""

# --- Guard: check previous kfold results ---
if [ -d "runs/kfold" ]; then
    echo "========================================================"
    echo "[Notice] Previous K-Fold results found in runs/kfold/."
    echo "  R = Resume  (skip completed folds, resume interrupted)"
    echo "  O = Overwrite (start all folds from scratch)"
    echo "  N = Cancel"
    echo "========================================================"
    read -p "Your choice (R/O/N, default is N): " kfold_action

    if [[ "$kfold_action" =~ ^[Rr]$ ]]; then
        KFOLD_EXTRA="--resume"
        echo "Resuming K-Fold from checkpoints..."
    elif [[ "$kfold_action" =~ ^[Oo]$ ]]; then
        KFOLD_EXTRA=""
        echo "Starting K-Fold from scratch (overwrite)..."
    else
        echo "K-Fold training cancelled by user."
        exit 0
    fi
    echo ""
fi

echo "========================================================"
echo "Starting YOLOv8 K-Fold Cross Validation..."
echo "========================================================"
# shellcheck disable=SC2086
python src/train_kfold.py --config config.yaml $KFOLD_EXTRA
if [ $? -ne 0 ]; then
    echo ""
    echo "[Error] K-Fold training encountered an error."
    exit 1
fi

echo ""
echo "========================================================"
echo "K-Fold Training Execution Finished."
echo "Results saved to: runs/kfold/"
echo "Weights saved to: weights/best_fold_1.pt ~ best_fold_N.pt"
echo "========================================================"
