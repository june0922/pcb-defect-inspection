# DeepPCB 데이터셋 기반 YOLOv8 PCB 결함 검출 및 판정 시스템

YOLOv8 객체 검출 모델을 활용하여 PCB(인쇄회로기판) 표면 결함을 자동 검출하고, 신뢰도 기반 3단계 판정(OK / NG / REVIEW)을 통해 **보드 단위 합격·불합격·수동 검토** 판정까지 수행하는 검사 시스템입니다.

> **핵심 차별점** — 단순 bounding box 검출에 그치지 않고, YOLO 추론 결과의 confidence를 구간별로 분류하여 **Recall 우선 판정** 로직을 구현합니다. 애매한 신뢰도 구간의 검출을 REVIEW로 분류하여 불량 놓침을 최소화합니다.

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [동작 개요](#동작-개요)
3. [데이터셋 — DeepPCB](#데이터셋--deeppcb)
4. [설정 파일 상세](#설정-파일-상세)
5. [소스 코드 상세](#소스-코드-상세)
6. [웹 데모 — FastAPI 기반 검사 서버](#웹-데모--fastapi-기반-검사-서버)
7. [웹 뷰어 — 모델 성능 비교 도구](#웹-뷰어--모델-성능-비교-도구)
8. [실행 스크립트](#실행-스크립트)
9. [실행 가이드](#실행-가이드)
10. [스모크 테스트](#스모크-테스트)

---

## 프로젝트 구조

```text
ati3_project/
├── config.yaml              # 전체 설정 (환경·학습·튜닝·평가·판정 파라미터)
├── data.yaml                # YOLO 데이터셋 정의 (6-class, 경로 플레이스홀더)
├── requirements.txt         # Python 의존성
├── install.bat / .sh        # 가상환경 생성 및 의존성 자동 설치 스크립트
├── GPU_SERVER_GUIDE.md      # GPU 서버 환경 셋업 가이드
├── GPU_SERVER_USER_GUIDE.md # 윈도우/Mac 사용자용 서버 접속 가이드
├── .gitignore               # 전처리 데이터·학습 산출물 제외
├── .gitattributes           # dataset.zip, *.pt → Git LFS 추적
│
├── src/                     # 핵심 소스 코드
│   ├── __init__.py
│   ├── utils.py             # config 로드 + 환경별 경로 분기
│   ├── preprocess.py        # DeepPCB → YOLO 포맷 변환 + 데이터 분할
│   ├── train.py             # YOLOv8 단일 학습 (신규/이어학습)
│   ├── train_kfold.py       # Stratified K-Fold 교차 검증
│   ├── tune.py              # 유전 알고리즘 기반 하이퍼파라미터 튜닝
│   ├── evaluate.py          # test 세트 mAP / Recall 평가
│   └── visualize.py         # EDA 시각화 + 결과 주석 이미지 생성
│
├── scripts/                 # 실행 보조 스크립트
│   ├── run_train.bat / .sh  # 대화형 학습 실행 (전처리 → 학습 → 평가)
│   ├── run_kfold.bat / .sh  # K-Fold 교차 검증 실행
│   ├── run_tune.bat / .sh   # 하이퍼파라미터 튜닝 실행
│   ├── reset_to_main.bat / .sh  # main 브랜치 완전 리셋
│   └── show_config.py       # config.yaml 테이블 출력
│
├── web_hwang/               # FastAPI 웹 데모 (단일 이미지 + 보드 격자 검사 API)
│   ├── app.py               
│   ├── pcb_inspect.py       # OK/NG/REVIEW 판정 로직
│   ├── visualize.py         
│   ├── tools/               
│   │   └── build_demo_boards.py  # 가상 보드 생성 스크립트
│   ├── static/              
│   ├── samples/             
│   └── README.md
│
├── web_test/                # 모델 튜닝 전후 결과 비교 및 정적 웹 뷰어
│   ├── generate_results.py  # 모델 추론 결과(WBF 앙상블 등) 및 mAP 생성
│   ├── index.html           # 정적 결과 뷰어
│   ├── script.js
│   └── style.css
│
├── weights/                 # 모델 가중치 (.pt)
├── dataset/                 # DeepPCB 원본 데이터 (dataset.zip 압축 해제)
├── preprocessed_data/       # YOLO 포맷 변환 결과 (전처리 후 생성)
├── runs/                    # YOLO 학습 로그 및 결과
└── notebooks/
    ├── pcb_train_colab.ipynb # Colab용 학습 노트북
    ├── pcb_tune_colab.ipynb  # Colab용 하이퍼파라미터 튜닝 노트북
    └── pcb_kfold_colab.ipynb # Colab용 K-Fold 검증 노트북
```

---

## 동작 개요

프로젝트 전체의 데이터 흐름은 다음과 같습니다.

```text
┌─────────────────┐
│  DeepPCB 원본   │  dataset/PCBData/
│  (640×640 이미지 │  ├── group00000~00099/
│   + 절대좌표     │  │   ├── *_test.jpg  (결함 이미지)
│     라벨)        │  │   ├── *_temp.jpg  (정상 템플릿)
└────────┬────────┘   │   └── *_not/*_test.txt (라벨)
         │
         │ src/preprocess.py
         ▼ DeepPCB 포맷 → YOLO 정규화 xywh 변환 + 70/20/10 split
┌─────────────────┐
│ preprocessed_   │  preprocessed_data/
│    data/        │  ├── images/{train,val,test}/*.jpg
│ (YOLO 포맷)     │  └── labels/{train,val,test}/*.txt
└────────┬────────┘
         │
         │ src/train_kfold.py
         ▼ YOLOv8 모델 학습
┌─────────────────┐
│  학습된 모델     │  weights/best_1~5.pt
│  (best_1~5.pt)  │  runs/train/  (로그, confusion matrix, 곡선 등)
└────────┬────────┘
         │
         ├── src/evaluate.py  →  test 세트 mAP / Recall 평가
         │
         ├── web_hwang/app.py  →  웹 데모 (단일 이미지 + 보드 격자 검사)
         │
         └── web_test/generate_results.py → (테스트용) 튜닝 전/후 WBF 앙상블 결과 비교용 데이터 생성
```

---

## 데이터셋 — DeepPCB

[DeepPCB](https://github.com/tangsanli5201/DeepPCB)는 PCB 결함 검출 연구를 위한 공개 데이터셋입니다.

### 원본 구조

```text
PCBData/
├── trainval.txt                    # 학습+검증 이미지-라벨 목록
├── test.txt                        # 테스트 이미지-라벨 목록
└── group00041/
    └── 00041/
        ├── 00041000_temp.jpg       # 결함 없는 정상 템플릿 이미지
        ├── 00041000_test.jpg       # 결함이 있는 테스트 이미지
        └── 00041000_not/
            └── 00041000_test.txt   # 결함 라벨 (x1 y1 x2 y2 type)
```

- **이미지 크기**: 640 × 640 px (모두 동일)
- **라벨 포맷**: `x1 y1 x2 y2 type` (공백 구분, 절대 픽셀 좌표, type 1~6)
- `_test.jpg` — 결함이 존재하는 실제 검사 이미지
- `_temp.jpg` — 결함이 없는 정상 템플릿 이미지

### 결함 클래스 (6종)

DeepPCB 원본 type 1~6을 YOLO cls 0~5로 매핑합니다 (`cls = type - 1`).

| YOLO cls | 클래스명     | 원본 type | 설명                               |
|:--------:|:----------:|:---------:|:----------------------------------|
| 0        | **open**     | 1         | 단선 — 회로 배선이 끊어진 결함            |
| 1        | **short**    | 2         | 단락 — 인접 배선이 연결된 결함            |
| 2        | **mousebite**| 3         | 마우스바이트 — 배선 가장자리가 불규칙하게 깎인 결함 |
| 3        | **spur**     | 4         | 돌기 — 배선에서 불필요한 돌출이 발생한 결함     |
| 4        | **copper**   | 5         | 동잔류 — 제거되어야 할 구리가 남아있는 결함     |
| 5        | **pinhole**  | 6         | 핀홀 — 배선에 미세한 구멍이 생긴 결함        |

---

## 설정 파일 상세

### `config.yaml` — 중앙 설정 파일

프로젝트의 모든 모듈이 이 파일을 읽어 동작합니다. 세부적인 하이퍼파라미터 및 증강(Augmentation) 옵션 등이 정의되어 있습니다.

```yaml
env: local                    # 실행 환경 (server / colab / local)

paths:                        # 환경별 경로 설정
  server:
    raw_data: dataset/PCBData
    project_root: "."
  colab:
    raw_data: dataset/PCBData
    project_root: /content/pcb-defect-inspection
  local:
    raw_data: dataset/PCBData
    project_root: "."

preprocess:                   # 전처리 설정
  img_size: 640               # 변환할 이미지 크기

split:                        # 데이터 분할 비율
  train: 0.7
  val: 0.2
  test: 0.1
  random_state: 42

train:                        # 학습 파라미터 및 하이퍼파라미터 증강 옵션
  model: weights/yolov8n.pt   # 베이스 모델
  epochs: 500                 
  batch: 128
  imgsz: 640
  patience: 50                # Early Stopping
  workers: 8
  device: 0                   # 0=GPU, cpu=CPU
  cache: disk                 # 디스크 캐싱 사용
  optimizer: AdamW            # 옵티마이저
  lr0: 0.001                  # 초기 학습률
  lrf: 0.01                   # 최종 학습률 비율
  cos_lr: true                # 코사인 학습률 감소
  # Augmentation 옵션
  flipud: 0.5
  fliplr: 0.5
  mosaic: 1.0
  # Loss 옵션
  box: 10.0
  cls: 1.0
  dfl: 2.0
  rect: true
  iou: 0.65

tune:                         # 하이퍼파라미터 튜닝 설정 (Train 설정과 유사)
  iterations: 20              # 튜닝 알고리즘 반복 횟수
  ...

kfold:                        # K-Fold 교차 검증
  k: 5
  random_state: 42

evaluate:                     # 평가 설정
  weights: weights/best.pt    # 평가에 사용할 가중치 파일

judge:                        # 판정 기준값
  conf_threshold: 0.5         # 이 이상이면 NG
  iou_threshold: 0.45
  review_band: [0.3, 0.5]     # 이 구간이면 REVIEW
```

| 환경    | `env` 값  | 용도 |
|:------:|:--------:|:----|
| 서버   | `server` | GPU 원격 서버에서 학습 실행 |
| 콜랩   | `colab`  | Google Colab 환경 |
| 로컬   | `local`  | 로컬 Windows/Mac/Linux 개발 환경 |

### `data.yaml` — YOLO 데이터셋 정의

```yaml
path: PLACEHOLDER_SET_BY_TRAIN   # 런타임에 train.py가 실제 절대 경로로 주입
train: images/train
val:   images/val
test:  images/test
nc: 6
names: [open, short, mousebite, spur, copper, pinhole]
```

> **참고**: `path` 필드는 `PLACEHOLDER_SET_BY_TRAIN`으로 설정되어 있으며, `train.py`·`evaluate.py`·`tune.py` 등이 실행 시 `build_data_yaml()` 함수를 통해 실제 절대 경로가 주입된 **임시 YAML 파일**을 생성하여 사용합니다. 이는 프로젝트 파일에 절대 경로가 커밋되는 것을 방지하기 위한 설계입니다.

---

## 소스 코드 상세

### `utils.py` — 환경 분기 유틸리티

모든 모듈이 공통으로 사용하는 설정 로딩 및 경로 관리 유틸리티입니다.

```python
load_config(path="config.yaml") → dict
```
`config.yaml`을 YAML 파서로 읽어 Python dict로 반환합니다.

```python
get_paths(cfg: dict) → dict[str, Path]
```
`cfg["env"]`에 따라 환경별 경로를 해석하고, 아래 키를 가진 dict를 반환합니다:

| 키 | 경로 | 설명 |
|:--|:-----|:----|
| `raw_data`     | `dataset/PCBData`     | DeepPCB 원본 (읽기 전용) |
| `project_root` | `.`                   | 프로젝트 루트 |
| `processed`    | `preprocessed_data/`  | YOLO 포맷 변환 데이터 |
| `weights`      | `weights/`            | 모델 가중치 저장 위치 |
| `runs`         | `runs/`               | 학습 로그/결과 |

`processed`, `weights`, `runs` 디렉터리는 존재하지 않으면 자동으로 생성됩니다.

---

### `preprocess.py` — 데이터 전처리

DeepPCB 원본 데이터를 YOLO 학습에 적합한 형식으로 변환합니다.

#### 실행 방법

```bash
python src/preprocess.py [--config config.yaml] [--limit N]
```

| 옵션 | 기본값 | 설명 |
|:----|:------|:----|
| `--config` | `config.yaml` | 설정 파일 경로 |
| `--limit N` | 전체 처리 | 최대 처리 샘플 수 (스모크 테스트: `--limit 50`) |

#### 라벨 변환 (`convert_label()`)

DeepPCB의 절대좌표 라벨을 YOLO 정규화 좌표로 변환합니다:

```text
DeepPCB: x1 y1 x2 y2 type    (절대 픽셀, type 1~6)
    ↓
YOLO:    cls cx cy w h        (정규화 0~1, cls 0~5)
```

변환 수식:
- `cls = type - 1`
- `cx = (x1 + x2) / 2 / img_width`
- `cy = (y1 + y2) / 2 / img_height`
- `w = (x2 - x1) / img_width`
- `h = (y2 - y1) / img_height`

---

### `train.py` — 모델 학습

YOLOv8 객체 검출 모델의 단일 학습을 수행합니다.

#### 실행 방법

```bash
# 신규 학습
python src/train.py [--config config.yaml]

# 이어학습 (중단된 학습 재개)
python src/train.py --resume
```

| 옵션 | 기본값 | 설명 |
|:----|:------|:----|
| `--config` | `config.yaml` | 설정 파일 경로 |
| `--resume` | (비활성) | `runs/train/weights/last.pt`에서 학습 재개 |

#### 학습 산출물

```text
runs/train/
├── weights/
│   ├── best.pt             # 검증 mAP 최고점의 가중치
│   └── last.pt             # 마지막 에폭 가중치 (이어학습용)
├── results.csv             # 에폭별 loss, mAP, Recall 수치
├── results.png             # 학습 곡선 그래프
├── confusion_matrix.png    # 혼동 행렬
└── args.yaml               # 실제 적용된 학습 인자 기록
```

---

### `train_kfold.py` — K-Fold 교차 검증

Stratified K-Fold 교차 검증으로 모델의 일반화 성능을 평가합니다. 희귀 클래스(예: pinhole) 기준으로 데이터가 분할되도록 층화(Stratified) 방식을 사용합니다.

#### 실행 방법

```bash
python src/train_kfold.py [--config config.yaml]
```

---

### `tune.py` — 하이퍼파라미터 튜닝

Ultralytics 내장 유전 알고리즘(Genetic Algorithm)으로 최적의 하이퍼파라미터를 자동 탐색합니다. `config.yaml`의 `tune` 섹션 파라터를 기반으로 동작합니다.

#### 실행 방법

```bash
python src/tune.py [--config config.yaml]
```

> **주의**: 튜닝은 `iterations × epochs` 만큼의 학습을 반복하므로 일반 학습보다 **매우 오래 걸립니다**.

---

### `evaluate.py` — 테스트 세트 평가

`config.yaml`의 `evaluate.weights`에 지정된 모델(기본: `weights/best.pt`)을 불러와 **test 세트** 성능을 정량적으로 측정합니다. 

#### 실행 방법

```bash
python src/evaluate.py [--config config.yaml]
```

제조 현장에서는 오탐(정상을 불량으로 판정)보다 **미탐(불량을 정상으로 통과)**이 훨씬 큰 손실을 유발하므로, 본 시스템은 불량을 얼마나 빠짐없이 검출하는지 보여주는 **Recall(재현율)**을 특히 중요하게 평가합니다.

---

### `pcb_inspect.py` — 보드 판정 로직

YOLO 검출 결과를 기반으로 **보드 단위 OK/NG/REVIEW 3단계 판정**을 수행하는 핵심 모듈입니다.

#### 실행 방법

```bash
# CLI 단일 이미지 판정
python web_hwang/pcb_inspect.py <이미지 경로>
```

#### 판정 로직 (Recall 우선 설계)

| 판정 | 조건 | 의미 |
|:---:|:----|:----|
| **OK** | 검출 없음, 또는 모든 검출이 `review_lower` 미만 | 양품 — 자동 통과 |
| **NG** | `conf ≥ conf_threshold` 인 검출 존재 | 불량 — 자동 불합격 |
| **REVIEW** | `review_lower ≤ conf < conf_threshold` 구간의 검출만 존재 | 수동 검토 — 사람이 최종 판정 |

기본 설정: `conf_threshold = 0.5`, `review_band = [0.3, 0.5]`

> **설계 철학**: `review_lower`를 낮게 잡아 애매한 검출을 절대 자동 통과시키지 않습니다. 노이즈 수준(< 0.3)만 무시합니다.

---

### `visualize.py` — EDA 및 결과 시각화

데이터 분석(EDA)과 검사 결과 이미지 생성(각 결함 및 bbox 랜더링) 기능을 제공합니다. 

---

## 웹 데모 — FastAPI 기반 검사 서버

단일 이미지 검사 및 보드 격자 검사를 웹 인터페이스에서 시연하기 위한 도구입니다 (`web_hwang` 디렉터리).

### 실행 방법

```bash
uvicorn web_hwang.app:app --reload --port 8000
```

접속 주소:
- `http://localhost:8000` — 단일 이미지 검사
- `http://localhost:8000/board` — 전체 보드 격자 검사

### 단일 이미지 검사 (`/`)
`POST /inspect` API를 호출하여 YOLO 추론, `inspect_image()` 판정, 시각화된 주석 이미지를 제공합니다.

### 전체 보드 격자 검사 (`/board`)
`web_hwang/tools/build_demo_boards.py` 스크립트로 생성한 16칸 합성 보드를 칸별 순차 검사하고, 칸별 집계를 통해 보드 전체 판정(OK/NG/REVIEW)을 내립니다.

---

## 웹 뷰어 — 모델 성능 비교 도구

모델 추론 결과를 테스트 데이터셋에 대해 직관적으로 시각화하고 mAP 수치를 비교하기 위한 정적 웹 뷰어입니다 (`web_test` 디렉터리). 
튜닝 전 모델과 튜닝 후 앙상블(WBF) 적용 모델의 탐지 결과를 한눈에 비교할 수 있습니다. 

### 실행 방법

1. **데이터 생성**:
   ```bash
   # 테스트 데이터셋을 모델로 추론하고 bounding box를 생성
   python web_test/generate_results.py
   ```
   이 스크립트는 모델 예측을 수행하고 mAP 계산 결과와 주석 이미지를 `web_test/results` 폴더 및 `data.js` 파일로 추출합니다.

2. **웹 뷰어 확인**:
   별도의 서버 띄울 필요 없이, 로컬 브라우저에서 `web_test/index.html` 파일을 직접 열어서 결과를 확인할 수 있습니다.

---

## 실행 스크립트

`scripts` 및 홈 디렉터리에 제공되는 스크립트를 통해 작업을 간편하게 수행할 수 있습니다.

### `install.bat` / `install.sh`
- **의존성 자동 셋업**: `venv` 가상환경을 만들고 `requirements.txt`에 명시된 파이썬 라이브러리들을 일괄 설치합니다. 

### `scripts/run_train.bat` / `run_train.sh` — 대화형 학습 실행
- 전처리 실행 여부 확인, 설정 로드, 학습 및 이어학습(resume) 등을 대화형 콘솔로 안내합니다.

### `scripts/run_kfold.bat` / `run_kfold.sh` 
- K-Fold 검증 학습을 자동화하는 스크립트.

### `scripts/run_tune.bat` / `run_tune.sh` 
- 하이퍼파라미터 튜닝을 실행하는 대화형 스크립트.

### `scripts/reset_to_main.bat` / `reset_to_main.sh`
- ⚠️ **파괴적 동작**: 모든 로컬 변경사항, 전처리 및 학습 파일들을 완전 삭제하고 `origin/main`으로 원상복구합니다.

---

## 실행 가이드

### 전체 실행 순서

```bash
# 0. 의존성 설치 (가상환경 생성 및 설치 자동화)
# 윈도우 사용자: install.bat / Linux&Mac 사용자: bash install.sh
install.bat

# 가상환경 활성화 (필수)
# 윈도우: venv\Scripts\activate / Linux&Mac: source venv/bin/activate

# 1. 전처리 (DeepPCB → YOLO 포맷 변환 + 70/20/10 split)
python src/preprocess.py

# 2. 모델 학습
# scripts/run_train.bat (대화형) 또는 직접 실행:
python src/train.py

# 3. K-Fold 교차 검증 (선택사항)
python src/train_kfold.py

# 4. 하이퍼파라미터 튜닝 (선택사항 — 매우 오래 걸림)
python src/tune.py

# 5. 테스트 세트 평가
python src/evaluate.py

# 6. 단일 이미지 판정
python web_hwang/pcb_inspect.py preprocessed_data/images/test/<이미지>.jpg

# 7. FastAPI 웹 데모 실행
uvicorn web_hwang.app:app --reload --port 8000

# 8. 모델 비교 뷰어용 결과 생성
python web_test/generate_results.py
# 이후 web_test/index.html 파일을 웹 브라우저에서 직접 오픈
```

> 💡 **Colab 사용자**: `notebooks/` 디렉터리에 있는 `pcb_train_colab.ipynb`, `pcb_tune_colab.ipynb`, `pcb_kfold_colab.ipynb`를 구글 드라이브나 Colab에 업로드하여 목적에 맞게 클라우드 환경에서 실행하실 수 있습니다.

---

## 스모크 테스트

신규 환경 셋업 후 전체 파이프라인이 정상 동작하는지 빠르게 확인합니다.

```bash
# 1. 의존성 설치
install.bat
venv\Scripts\activate

# 2. config.yaml 에서 env 설정 확인 (server / colab / local)

# 3. 전처리 — 50장만 처리 (빠른 검증)
python src/preprocess.py --limit 50
# 기대: preprocessed_data/images/{train,val,test} 디렉터리 생성

# 4. 스모크 학습 (config.yaml에서 epochs를 1로 임시 변경 후)
python src/train.py
# 기대: runs/train/ 생성, weights/best.pt 복사 완료

# 5. 평가
python src/evaluate.py
# 기대: Recall / mAP@0.5 / mAP@0.5:0.95 수치 출력

# 6. 단일 이미지 판정
python web_hwang/pcb_inspect.py preprocessed_data/images/test/<아무_이미지>.jpg
# 기대: 판정(OK/NG/REVIEW) + 결함 목록 출력

# 7. 웹 데모 확인
uvicorn web_hwang.app:app --reload --port 8000
# 기대: http://localhost:8000 접속 후 검사 동작 확인
```
