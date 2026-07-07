#!/bin/bash
# YOLOv8 Hyperparameter Tuning Script

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment if it exists
if [ -f "$(dirname "$0")/../venv/bin/activate" ]; then
    source "$(dirname "$0")/../venv/bin/activate"
fi

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
BASE_MODEL=$(python scripts/get_base_model.py)
if [ ! -f "$BASE_MODEL" ]; then
    echo "[Warning] $BASE_MODEL not found."
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
    echo "  R = Resume  (continue from last completed iteration)"
    echo "  O = Overwrite (delete results and start from scratch)"
    echo "  N = Cancel (keep existing results and exit)"
    echo "========================================================"
    read -p "Your choice (R/O/N, default is N): " tune_action

    if [[ "$tune_action" =~ ^[Rr]$ ]]; then
        echo "Resuming previous tuning run..."
    elif [[ "$tune_action" =~ ^[Oo]$ ]]; then
        echo "Deleting previous tuning results and starting from scratch..."
        rm -rf "runs/tune"
    else
        echo "Tuning cancelled by user."
        exit 0
    fi
    echo ""
fi

echo "========================================================"
echo "Starting YOLOv8 Hyperparameter Tuning..."
echo "[Note] This process runs multiple short training cycles."
echo "       tune.py auto-detects runs/tune/tune_results.ndjson to resume or start fresh."
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

if [ -n "$TMUX" ]; then
    echo "Running in tmux. The session will automatically close in 10 seconds."
    echo "Press Ctrl+C to cancel and keep the session open."
    sleep 10
    tmux kill-session
else
    echo "The script will exit in 10 seconds. Press Ctrl+C to cancel."
    sleep 10
fi
