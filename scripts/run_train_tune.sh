#!/bin/bash
# YOLOv8 Model Training Script (Hyperparameter tuned)

# Change working directory to the project root
cd "$(dirname "$0")/.." || exit

# Activate virtual environment if it exists
if [ -f "$(dirname "$0")/../venv/bin/activate" ]; then
    source "$(dirname "$0")/../venv/bin/activate"
fi

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

TRAIN_CMD="python src/train_tune.py --config config.yaml"
if [ -f "runs/train_tune/weights/last.pt" ]; then
    echo "========================================================"
    echo "[Notice] Found a previous training checkpoint (last.pt)."
    echo "========================================================"
    read -p "Do you want to resume training from this checkpoint? (Y/N, default is N): " do_resume
    if [[ "$do_resume" =~ ^[Yy]$ ]]; then
        TRAIN_CMD="python src/train_tune.py --config config.yaml --resume"
    fi
    echo ""
fi

echo "========================================================"
echo "Starting YOLOv8 Model Training (train_tune)..."
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

if [ -n "$TMUX" ]; then
    echo "Running in tmux. The session will automatically close in 10 seconds."
    echo "Press Ctrl+C to cancel and keep the session open."
    sleep 10
    tmux kill-session
else
    echo "The script will exit in 10 seconds. Press Ctrl+C to cancel."
    sleep 10
fi
