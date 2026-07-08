#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "[INFO] 가상환경이 없습니다. 현재 Python 환경을 사용합니다."
fi

echo "============================================================"
echo " DeepPCB PCB 결함 검사 시스템"
echo " (Review Station + Inspection Monitor 동시 실행)"
echo "============================================================"
echo ""
echo "[1/2] app_back (Review Station) 시작 중..."
echo "      - 5개 YOLO 모델 로딩에 약 30초~1분 소요됩니다."
python app_back/run.py &
BACK_PID=$!

echo ""
echo "[대기] app_back 초기화를 위해 3초 대기 중..."
sleep 3

echo ""
echo "[2/2] app_front (Inspection Monitor) 시작 중..."
python app_front/run.py &
FRONT_PID=$!

echo ""
echo "============================================================"
echo " 두 프로그램이 시작되었습니다."
echo " - Review Station (PID: $BACK_PID): REVIEW 타일 수신 대기 중"
echo " - Inspection Monitor (PID: $FRONT_PID): 폴더 선택 후 검사 시작"
echo "   (File > Open Folder...)"
echo "============================================================"

wait
