#!/bin/bash
cd "$(dirname "$0")"

echo "========================================"
echo "Starting Jigsaw Reconstruction Pipeline"
echo "========================================"

if [ -z "$1" ]; then
    TARGET_GROUP="all"
    echo "No group specified, defaulting to 'all' (processes all groups)"
else
    TARGET_GROUP="$1"
    echo "Target group: $1"
fi

echo ""
echo "Running Jigsaw Solver..."
python3 ../src/jigsaw_solver.py --group "$TARGET_GROUP"
if [ $? -ne 0 ]; then
    echo "[ERROR] Jigsaw Solver failed."
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "========================================"
echo "Jigsaw Reconstruction Completed successfully."
echo "Please check the recovered_data directory."
echo "========================================"
read -p "Press Enter to continue..."
exit 0
