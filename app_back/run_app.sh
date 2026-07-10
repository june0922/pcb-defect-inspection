#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "[INFO] 가상환경이 없습니다. 현재 Python 환경을 사용합니다."
fi

echo "============================================================"
echo " DeepPCB Review Station (app_back)"
echo "============================================================"
echo ""
python app_back/run.py
