#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -f "$(dirname "$0")/../venv/bin/activate" ]; then
    source "$(dirname "$0")/../venv/bin/activate"
fi

echo "========================================"
echo " DeepPCB Defect Review Station"
echo "========================================"
echo

echo "[INFO] Starting review station..."
echo

python app/run.py
if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Application exited with error."
    exit 1
fi

exit 0
