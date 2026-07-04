#!/bin/bash
cd "$(dirname "$0")"

echo "======== PRE-FLIGHT CHECKS ========"
# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    echo "Please install Python 3 and try again."
    exit 1
fi
echo "[INFO] Python 3 is installed."

echo "======== VIRTUAL ENVIRONMENT ========"
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
    echo "[INFO] Virtual environment created successfully."
else
    echo "[INFO] Virtual environment 'venv' already exists."
fi

echo "======== INSTALLING DEPENDENCIES ========"
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip..."
python3 -m pip install --upgrade pip > /dev/null

echo "[INFO] Installing packages from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found."
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements."
    exit 1
fi

echo "======== SUCCESS ========"
echo "[INFO] All requirements have been installed successfully."
echo "[INFO] To activate the environment manually in the future, run:"
echo "       source venv/bin/activate"
echo "========================="
exit 0
