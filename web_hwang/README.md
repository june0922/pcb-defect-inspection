# PCB 결함 검사 웹 데모

Team Convex 인턴 프로젝트 — 발표 시연용 로컬 데모.  
`web_hwang/pcb_inspect.py` 의 OK / NG / REVIEW 판정 로직을 사용하는 웹 래퍼입니다.

---

## 준비

### 1. best.pt 모델 파일

`best.pt`는 `.gitignore`에 등록되어 git에 포함되지 않습니다.  
아래 두 경로 중 하나에 두면 서버가 자동으로 찾습니다 (위에서부터 우선 탐색). **만약 두 경로 모두에 `best.pt`가 없으면, 사전 학습된 `yolov8s.pt`를 자동으로 다운로드하여 폴백 모드로 실행됩니다.**

```
pcb-project/
├── web_hwang/
│   └── best.pt          ← 우선 탐색 (데모용 권장)
└── weights/
    └── best.pt          ← 차선 탐색 (학습 후 자동 저장 위치)
```

### 2. 샘플 이미지 및 보드 준비

- **단일 검사 이미지**: `web_hwang/samples/` 에 시연용 단일 이미지 7장을 넣어두면 좌측 썸네일로 표시됩니다.
- **가상 보드 이미지**: 전체 보드 검사 시연을 위해 가상 보드를 생성할 수 있습니다. 레포지토리 루트에서 아래 스크립트를 실행하면 `web_hwang/samples/boards/` 에 가상 보드가 3종(`ok_board`, `ng_board`, `review_board`) 생성됩니다.
  ```bash
  python web_hwang/tools/build_demo_boards.py --raw-data /path/to/DeepPCB/PCBData --group group00041
  ```

---

## 실행

```bash
# 1. 레포 루트로 이동
cd /path/to/pcb-project

# 2. 의존성 설치 (최초 1회)
pip install -r requirements.txt

# 3. best.pt 를 web_hwang/ 또는 weights/ 에 배치

# 4. 서버 시작
uvicorn web_hwang.app:app --reload --port 8000

# 5. 브라우저에서 접속
# → 단일 결함 검사: http://localhost:8000
# → 전체 보드 격자 검사: http://localhost:8000/board
```

> `--reload` 는 코드 변경 시 자동 재시작. 발표 당일에는 빼도 됩니다.

---

## 화면 구성 및 기능

데모는 **단일 이미지 검사**와 **보드 격자 검사** 두 가지 모드를 지원합니다. `config.yaml`의 `judge` 설정값(예: `conf_threshold`, `review_band`)에 따라 판정(Recall 우선)이 이루어집니다.

### 1. 단일 이미지 검사 (`/`)
좌측에서 샘플 썸네일을 클릭하거나 직접 이미지를 업로드하면 결과를 즉시 확인합니다.
- **판정 기준 (`web_hwang/pcb_inspect.py`)**:
  - **OK** (초록): 발견된 결함이 0개이거나 노이즈 수준일 때
  - **NG** (빨강): `conf_threshold` 이상 신뢰도의 확실한 결함이 존재할 때
  - **REVIEW** (주황): `review_band` 구간의 애매한 결함만 존재하여 수동 검토가 필요할 때
- **지원 결함 클래스**: open, short, mousebite, spur, copper, pinhole

```
┌──────────┬─────────────────────┬──────────────┐
│ 검사 보드  │   주석 이미지          │  판정 결과    │
│ 선택      │                     │              │
│ [thumb]  │  ┌───────────────┐  │  [ OK ]      │
│ [thumb]  │  │  bbox 결함 표시│  │  결함: 0개    │
│ [thumb]  │  └───────────────┘  │  클래스별 수  │
│ ...      │                     │  결함 목록    │
│ ↑업로드  │                     │  판정 기준    │
└──────────┴─────────────────────┴──────────────┘
```

### 2. 보드 단위 검사 (`/board`)
`build_demo_boards.py`로 합성된 대형 PCB 보드를 격자(예: 4×4)로 분할하여 칸별로 검사하고, 보드 전체 판정을 종합하여 보여줍니다.
- **보드 전체 판정 기준 (`web_hwang/app.py`)**:
  - 격자 중 `NG` 칸이 하나라도 있으면 → 보드 **NG**
  - `NG` 칸은 없고 `REVIEW` 칸이 있으면 → 보드 **REVIEW**
  - 모든 격자 칸이 `OK` → 보드 **OK**

---

## 발표 전 체크리스트

```
[ ] best.pt 를 web_hwang/ 또는 weights/ 에 배치 확인
[ ] web_hwang/samples/ 에 단일 검사용 샘플 7장 준비
[ ] python web_hwang/tools/build_demo_boards.py 실행하여 가상 보드 3종 생성
[ ] pip install -r requirements.txt 완료
[ ] uvicorn web_hwang.app:app --port 8000 실행 → 콘솔에 "워밍업 완료" 확인
[ ] 브라우저에서 http://localhost:8000 (단일 검사), http://localhost:8000/board (보드 검사) 리허설
[ ] 업로드 기능도 백업으로 1회 확인
```

---

## 파일 구조

```
web_hwang/
├── app.py              # FastAPI 서버 (모델 로드·라우팅, 단일/보드 검사 API)
├── pcb_inspect.py      # OK / NG / REVIEW 판정 핵심 로직
├── visualize.py        # 결과 이미지 시각화 (BBox, 라벨 렌더링)
├── tools/
│   └── build_demo_boards.py # 데모용 가상 보드 조립 스크립트
├── static/
│   ├── index.html      # 단일 검사 화면 (3패널 레이아웃)
│   ├── board.html      # 전체 보드 격자 검사 화면
│   ├── style.css       # 공장 단말 스타일 (다크 테마)
│   └── script.js       # 샘플 클릭·업로드·결과 렌더링
├── samples/            # 시연용 단일 이미지 (git 포함)
│   └── boards/         # 생성된 가상 보드 이미지 저장 경로
└── README.md           # 이 파일
```
