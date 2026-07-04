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
   - [utils.py — 환경 분기 유틸리티](#utilspy--환경-분기-유틸리티)
   - [preprocess.py — 데이터 전처리](#preprocesspy--데이터-전처리)
   - [train.py — 모델 학습](#trainpy--모델-학습)
   - [train_kfold.py — K-Fold 교차 검증](#train_kfoldpy--k-fold-교차-검증)
   - [tune.py — 하이퍼파라미터 튜닝](#tunepy--하이퍼파라미터-튜닝)
   - [evaluate.py — 테스트 세트 평가](#evaluatepy--테스트-세트-평가)
   - [pcb_inspect.py — 보드 판정 로직](#pcb_inspectpy--보드-판정-로직)
   - [visualize.py — EDA 및 결과 시각화](#visualizepy--eda-및-결과-시각화)
6. [웹 데모 — FastAPI 기반 검사 서버](#웹-데모--fastapi-기반-검사-서버)
7. [실행 스크립트](#실행-스크립트)
8. [실행 가이드](#실행-가이드)
9. [스모크 테스트](#스모크-테스트)
10. [브랜치 분담](#브랜치-분담)

---

## 프로젝트 구조

```
ati3_project/
├── config.yaml              # 전체 설정 (환경·학습·판정 파라미터)
├── data.yaml                # YOLO 데이터셋 정의 (6-class, 경로 플레이스홀더)
├── requirements.txt         # Python 의존성
├── GPU_SERVER_GUIDE.md      # GPU 서버 환경 셋업 가이드
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
│   ├── pcb_inspect.py       # 단일 이미지 OK/NG/REVIEW 판정
│   └── visualize.py         # EDA 시각화 + 결과 주석 이미지 생성
│
├── scripts/                 # 실행 보조 스크립트
│   ├── run_train.bat / .sh  # 대화형 학습 실행 (전처리 → 학습 → 평가)
│   ├── reset_to_main.bat / .sh  # main 브랜치 완전 리셋
│   └── show_config.py       # config.yaml 테이블 출력
│
├── web_hwang/                     # FastAPI 웹 데모
│   ├── app.py               # 서버 (단일 이미지 + 보드 격자 검사 API)
│   ├── static/              # 프론트엔드 (HTML / CSS / JS)
│   │   ├── index.html       # 단일 이미지 검사 페이지
│   │   ├── board.html       # 전체 보드 격자 검사 페이지
│   │   ├── style.css / board.css
│   │   ├── script.js / board.js
│   ├── samples/             # 데모용 샘플 이미지 + 가상 보드
│   └── tools/
│       └── build_demo_boards.py  # 가상 보드 생성 스크립트
│
├── weights/                 # 모델 가중치 (.pt)
├── dataset/                 # DeepPCB 원본 데이터 (dataset.zip 압축 해제)
├── preprocessed_data/       # YOLO 포맷 변환 결과 (전처리 후 생성)
├── runs/                    # YOLO 학습 로그 및 결과
└── notebooks/
    └── pcb_colab_verify.ipynb  # Colab 검증 노트북
```

---

## 동작 개요

프로젝트 전체의 데이터 흐름은 다음과 같습니다.

```
┌─────────────────┐
│  DeepPCB 원본   │  dataset/PCBData/
│  (640×640 이미지 │  ├── group00000~00099/
│   + 절대좌표     │  │   ├── *_test.jpg  (결함 이미지)
│     라벨)        │  │   ├── *_temp.jpg  (정상 템플릿)
└────────┬────────┘  │   └── *_not/*_test.txt (라벨)
         │
         │ src/preprocess.py
         ▼ DeepPCB 포맷 → YOLO 정규화 xywh 변환 + 70/20/10 split
┌─────────────────┐
│ preprocessed_   │  preprocessed_data/
│    data/        │  ├── images/{train,val,test}/*.jpg
│ (YOLO 포맷)     │  └── labels/{train,val,test}/*.txt
└────────┬────────┘
         │
         │ src/train.py (또는 train_kfold.py, tune.py)
         ▼ YOLOv8 모델 학습
┌─────────────────┐
│  학습된 모델     │  weights/best.pt
│  (best.pt)      │  runs/train/  (로그, confusion matrix, 곡선 등)
└────────┬────────┘
         │
         ├── src/evaluate.py  →  test 세트 mAP / Recall 평가
         │
         ├── src/pcb_inspect.py  →  단일 이미지 OK/NG/REVIEW 판정
         │
         └── web_hwang/app.py  →  웹 데모 (단일 이미지 + 보드 격자 검사)
```

---

## 데이터셋 — DeepPCB

[DeepPCB](https://github.com/tangsanli5201/DeepPCB)는 PCB 결함 검출 연구를 위한 공개 데이터셋입니다.

### 원본 구조

```
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

프로젝트의 모든 모듈이 이 파일을 읽어 동작합니다.

```yaml
env: local                    # 실행 환경 (server / colab / local)

paths:                        # 환경별 경로 설정
  server:
    raw_data: dataset/PCBData
    project_root: "."
  colab:
    raw_data: dataset/PCBData
    project_root: /content/pcb-project
  local:
    raw_data: dataset/PCBData
    project_root: "."

split:                        # 데이터 분할 비율
  train: 0.7
  val: 0.2
  test: 0.1
  random_state: 42

train:                        # 학습 하이퍼파라미터
  model: weights/yolov8n.pt   # 베이스 모델 (n → s → m 으로 스케일업 가능)
  epochs: 100
  batch: 16
  imgsz: 640
  patience: 50                # Early Stopping patience
  workers: 4

tune:                         # 하이퍼파라미터 튜닝
  model: weights/yolov8n.pt
  epochs: 30
  iterations: 30
  imgsz: 640
  workers: 4

kfold:                        # K-Fold 교차 검증
  k: 5
  random_state: 42

judge:                        # 판정 기준값
  conf_threshold: 0.5         # 이 이상이면 NG
  iou_threshold: 0.45
  review_band: [0.3, 0.5]    # 이 구간이면 REVIEW
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

#### 처리 흐름

```
1. prepare_dataset()
   └── dataset.zip 존재 시 자동 압축 해제

2. collect_pairs(raw_data, limit)
   ├── trainval.txt, test.txt 파일 파싱
   ├── 각 줄의 stem에서 실제 파일명 복원 (stem → stem_test.jpg)
   └── (image_path, label_path) 쌍 리스트 반환

3. split_dataset(pairs, train=0.7, val=0.2, random_state=42)
   ├── sklearn train_test_split 으로 2단계 분할
   └── train / val / test 리스트 반환

4. save_yolo_format(split_name, pairs, processed, cfg)  ← 3개 split 각각 호출
   ├── 이미지 파일 복사 → preprocessed_data/images/{split}/
   ├── cv2로 실제 이미지 크기 확인 (w, h)
   └── convert_label() → YOLO 포맷 라벨 저장
```

#### 라벨 변환 (`convert_label()`)

DeepPCB의 절대좌표 라벨을 YOLO 정규화 좌표로 변환합니다:

```
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

#### 출력

```
preprocessed_data/
├── images/
│   ├── train/   (전체의 70%)
│   ├── val/     (전체의 20%)
│   └── test/    (전체의 10%)
└── labels/
    ├── train/   (YOLO 포맷 .txt)
    ├── val/
    └── test/
```

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

#### 처리 흐름

```
1. config.yaml 로드 → 학습 파라미터 읽기

2. build_data_yaml()
   └── data.yaml의 path 플레이스홀더에 실제 preprocessed_data 절대 경로 주입
   └── 임시 YAML 파일 생성 (학습 완료 후 삭제)

3. 모델 로드
   ├── 신규: config의 model (기본 yolov8n.pt)
   └── resume: runs/train/weights/last.pt

4. model.train(data, epochs, batch, imgsz, patience, workers, ...)
   ├── Ultralytics 내장 학습 루프 실행
   ├── Early Stopping: patience=50 (50 에폭 동안 개선 없으면 자동 중단)
   └── 결과: runs/train/ 에 저장

5. best.pt 복사: runs/train/weights/best.pt → weights/best.pt
```

#### 학습 원리 — YOLOv8 모델 구조

YOLOv8은 단일 신경망이 **이미지 전체를 한 번 통과(Single Shot)**하면서 모든 객체의 위치와 클래스를 동시에 예측하는 객체 검출 모델입니다.

| 구성 요소 | 설명 |
|:---------|:----|
| **Backbone** (CSPDarknet) | 이미지에서 다양한 스케일의 특징(feature)을 추출합니다 |
| **Neck** (PANet/FPN) | 다중 스케일 특징을 융합하여 크고 작은 결함을 모두 감지합니다 |
| **Head** (Decoupled Head) | 분류(클래스)와 회귀(바운딩 박스)를 분리하여 예측합니다 |
| **Loss** | 분류 BCE Loss + 박스 회귀 CIoU Loss + DFL의 가중 합 |
| **NMS** | 중복 검출을 제거하여 최종 bounding box를 출력합니다 |

모델 크기 선택 (`config.yaml`의 `train.model`):

| 모델 | 파라미터 수 | 속도 | 정확도 | 권장 용도 |
|:----|:---------|:-----|:------|:---------|
| `yolov8n.pt` | 3.2M | 빠름 | 기본 | 빠른 실험·스모크 테스트 |
| `yolov8s.pt` | 11.2M | 보통 | 양호 | 성능-속도 균형 |
| `yolov8m.pt` | 25.9M | 느림 | 높음 | 최종 성능 검증 |

#### 학습 산출물

```
runs/train/
├── weights/
│   ├── best.pt             # 검증 mAP 최고점의 가중치
│   └── last.pt             # 마지막 에폭 가중치 (이어학습용)
├── results.csv             # 에폭별 loss, mAP, Recall 수치
├── results.png             # 학습 곡선 그래프
├── confusion_matrix.png    # 혼동 행렬
├── F1_curve.png            # F1-Confidence 곡선
├── P_curve.png             # Precision-Confidence 곡선
├── R_curve.png             # Recall-Confidence 곡선
├── PR_curve.png            # Precision-Recall 곡선
└── args.yaml               # 실제 적용된 학습 인자 기록
```

---

### `train_kfold.py` — K-Fold 교차 검증

Stratified K-Fold 교차 검증으로 모델의 일반화 성능을 평가합니다.

#### 실행 방법

```bash
python src/train_kfold.py [--config config.yaml]
```

#### 처리 흐름

```
1. preprocessed_data 에서 train + val 이미지를 하나의 풀로 통합

2. 층화(Stratified) 기준 결정
   └── get_representative_classes(): 각 이미지의 라벨 중
       전체 데이터에서 가장 희귀한 클래스를 대표 클래스로 지정
       → 소수 클래스가 모든 fold에 균등하게 분배됨

3. StratifiedKFold(n_splits=5, shuffle=True) 로 분할

4. Fold별 반복 (k=5):
   ├── fold용 train/val 이미지 경로 .txt 파일 생성
   ├── fold용 임시 data.yaml 생성
   ├── YOLO(model).train(data=fold_yaml, ...)
   └── best.pt → weights/best_fold_{k}.pt 백업

5. 전체 fold 완료 후 요약 출력
```

#### 왜 K-Fold를 사용하는가?

- **목적**: 단일 학습 결과가 데이터 분할 운에 의한 것인지 확인합니다
- **Stratified**: 각 fold에 모든 결함 클래스가 균등하게 포함됩니다
- **희귀 클래스 기준 층화**: 단순 최빈 클래스가 아닌, 전체 빈도가 가장 낮은 클래스를 대표로 사용하여 소수 클래스(예: pinhole)의 균등 분배를 보장합니다

#### 출력

```
runs/kfold/
├── fold_0/   (학습 결과)
├── fold_1/
├── ...
└── fold_4/

weights/
├── best_fold_0.pt
├── best_fold_1.pt
├── ...
└── best_fold_4.pt
```

---

### `tune.py` — 하이퍼파라미터 튜닝

Ultralytics 내장 유전 알고리즘(Genetic Algorithm)으로 최적의 하이퍼파라미터를 자동 탐색합니다.

#### 실행 방법

```bash
python src/tune.py [--config config.yaml]
```

> **주의**: 튜닝은 `iterations × epochs` 만큼의 학습을 반복하므로 일반 학습보다 **매우 오래 걸립니다** (기본 설정: 30회 반복 × 30 에폭 = 총 900 에폭).

#### 동작 원리

```
1. 초기 하이퍼파라미터 세트로 학습 시작 (1세대)

2. 이전 세대의 결과(mAP)를 기반으로 돌연변이(mutation) 적용
   탐색 대상: lr0, lrf, momentum, weight_decay,
              warmup_epochs, mosaic, flipud, fliplr,
              hsv_h, hsv_s, hsv_v, degrees, translate, scale 등

3. iterations 횟수만큼 반복하며 가장 높은 mAP를 달성한 파라미터 보존

4. 최적 파라미터를 best_hyperparameters.yaml 로 저장
```

#### 출력

```
runs/tune/
├── best_hyperparameters.yaml   # 최적 파라미터 (train.py에서 재사용 가능)
└── (각 iteration의 학습 로그)
```

---

### `evaluate.py` — 테스트 세트 평가

학습된 모델의 **test 세트** 성능을 정량적으로 측정합니다.

#### 실행 방법

```bash
python src/evaluate.py [--config config.yaml]
```

#### 처리 흐름

```
1. weights/best.pt 로드

2. model.val(data=data.yaml, split="test")
   └── test 세트 전체에 대해 추론 + 정답 비교

3. 핵심 메트릭 출력 (Recall 강조):
   ├── Recall (mean)     ← 불량 놓침률의 역수 (가장 중요)
   ├── mAP@0.5           ← IoU 50% 기준 평균 정밀도
   └── mAP@0.5:0.95      ← IoU 50~95% 엄격 기준
```

#### 왜 Recall을 강조하는가?

PCB 검사의 핵심 목표는 **불량을 놓치지 않는 것**입니다:
- **Recall이 높다** = 실제 결함을 대부분 검출했다 (놓침 적음)
- **Precision이 높다** = 오탐이 적다 (불필요한 경보 적음)

제조 현장에서는 오탐(정상을 불량으로 판정)보다 **미탐(불량을 정상으로 통과)**이 훨씬 큰 손실을 유발하므로, 본 시스템은 Recall 우선으로 설계되었습니다.

#### 출력

```
runs/eval/
├── confusion_matrix.png
└── (Ultralytics 표준 검증 결과물)
```

---

### `pcb_inspect.py` — 보드 판정 로직

YOLO 검출 결과를 기반으로 **보드 단위 OK/NG/REVIEW 3단계 판정**을 수행하는 핵심 모듈입니다.

#### 실행 방법

```bash
# CLI 단일 이미지 판정
python src/pcb_inspect.py <이미지 경로>

# 코드에서 함수 호출
from src.pcb_inspect import inspect_image
result = inspect_image("sample.jpg", model, cfg)
```

#### 판정 로직 (Recall 우선 설계)

```
                YOLO 추론 (conf ≥ review_lower 결과만 수집)
                             │
                    ┌────────┴────────┐
                    │ 검출 결과 있음?   │
                    └────────┬────────┘
                        No ──┤──── Yes
                        │         │
                    [ OK ]    ┌───┴───────────────────┐
                              │ conf ≥ conf_threshold  │
                              │ 인 결함이 있는가?       │
                              └───┬───────────────────┘
                             Yes ─┤─── No
                             │         │
                         [ NG ]    ┌───┴───────────────────┐
                                   │ review_band 구간 안    │
                                   │ 결함이 있는가?         │
                                   └───┬───────────────────┘
                                  Yes ─┤─── No
                                  │         │
                              [REVIEW]  [ OK ]  ← conf < review_lower 는 노이즈
```

| 판정 | 조건 | 의미 |
|:---:|:----|:----|
| **OK** | 검출 없음, 또는 모든 검출이 `review_lower` 미만 | 양품 — 자동 통과 |
| **NG** | `conf ≥ conf_threshold` 인 검출 존재 | 불량 — 자동 불합격 |
| **REVIEW** | `review_lower ≤ conf < conf_threshold` 구간의 검출만 존재 | 수동 검토 — 사람이 최종 판정 |

기본 설정: `conf_threshold = 0.5`, `review_band = [0.3, 0.5]`

> **설계 철학**: `review_lower`를 낮게 잡아 애매한 검출을 절대 자동 통과시키지 않습니다. 노이즈 수준(< 0.3)만 무시합니다.

#### 핵심 함수

| 함수 | 입력 | 출력 | 설명 |
|:----|:-----|:----|:----|
| `extract_defects(results)` | YOLO Results 객체 | `list[dict]` | YOLO 결과에서 결함 정보 추출 (class_id, class_name, conf, bbox, center) |
| `judge(defects, cfg)` | 결함 리스트, config | `(verdict, review_items)` | 결함 목록에 신뢰도 기반 판정 적용 |
| `inspect_image(image_path, model, cfg)` | 이미지 경로, YOLO 모델, config | `dict` | 추론 + 판정을 통합한 최종 검사 결과 반환 |

#### `inspect_image()` 반환값

```python
{
    "verdict": "OK" | "NG" | "REVIEW",
    "defect_count": 3,
    "by_class": {"open": 1, "short": 2},
    "defects": [
        {"class_id": 0, "class_name": "open", "conf": 0.85,
         "bbox": [100, 200, 150, 250], "center": (125.0, 225.0)},
        ...
    ],
    "review": [...]  # REVIEW 판정 시 애매한 결함 목록
}
```

---

### `visualize.py` — EDA 및 결과 시각화

데이터 분석(EDA)과 검사 결과 이미지 생성 기능을 제공합니다.

#### 주요 함수

| 함수 | 용도 | 입력 | 출력 |
|:----|:----|:----|:----|
| `plot_class_distribution(label_dir, save_path)` | 클래스별 결함 빈도 막대그래프 | YOLO 라벨 디렉터리 | 막대 그래프 이미지 |
| `plot_bbox_histogram(label_dir, img_size, save_path)` | bbox 너비/높이 분포 히스토그램 | YOLO 라벨 디렉터리 | 히스토그램 이미지 |
| `draw_inspection_result(image_path, inspection, save_path)` | 판정 결과를 이미지 위에 시각화 | 원본 이미지 + inspect_image() 결과 | 주석이 그려진 이미지 |

#### `draw_inspection_result()` 시각화 규칙

- **OK** → 초록색 (0, 200, 0)
- **NG** → 빨강색 (0, 0, 220)
- **REVIEW** → 주황색 (0, 140, 255)
- 이미지 상단에 판정 결과 배너 표시
- 각 결함에 클래스명 + 신뢰도 텍스트 표시

---

## 웹 데모 — FastAPI 기반 검사 서버

### 실행 방법

```bash
# 의존성 설치 (requirements.txt에 포함)
pip install -r requirements.txt

# 서버 시작 (프로젝트 루트에서)
uvicorn web_hwang.app:app --reload --port 8000
```

접속 주소:
- `http://localhost:8000` — 단일 이미지 검사
- `http://localhost:8000/board` — 전체 보드 격자 검사

### 서버 구조

서버 시작 시 `lifespan` 이벤트에서 다음을 수행합니다:
1. `config.yaml` 로드
2. 모델 탐색: `web_hwang/best.pt` → `weights/best.pt` → fallback `yolov8n.pt`
3. YOLO 모델 로드 + 더미 이미지 워밍업 (첫 요청 지연 방지)

### API 엔드포인트

| 메서드 | 경로 | 설명 |
|:------|:-----|:----|
| `GET /` | 단일 이미지 검사 페이지 서빙 |
| `GET /board` | 전체 보드 격자 검사 페이지 서빙 |
| `GET /samples` | 샘플 이미지 파일 목록 JSON 반환 |
| `GET /boards` | 가상 보드 목록 JSON 반환 |
| `GET /judge-config` | 판정 기준값 (conf_threshold, review_band) JSON 반환 |
| `POST /inspect` | 단일 이미지 추론 + 판정 + 주석 이미지(base64) 반환 |
| `POST /inspect_board` | 보드 격자 분할 → 칸별 추론 → 보드 전체 판정 반환 |
| `GET /board_cell/{board_id}/{row}/{col}` | 보드의 특정 칸 이미지(JPEG) 반환 |

### 단일 이미지 검사 (`/`)

1. 샘플 이미지 선택 또는 직접 업로드
2. `POST /inspect` 호출 → YOLO 추론 + `inspect_image()` 판정 + `draw_inspection_result()` 주석 이미지 생성
3. 결과: 판정 배지(OK/NG/REVIEW), 결함 수, 클래스별 집계, 결함 상세 테이블, 주석 이미지

### 전체 보드 격자 검사 (`/board`)

DeepPCB 640×640 이미지를 4×4 격자로 합성한 **데모용 가상 보드** 3종으로 검사 시나리오를 시연합니다.

| 보드 | 구성 | 기대 판정 |
|:----|:-----|:---------|
| `ok_board` | `_temp` 이미지 ×16 (정상 템플릿) | OK |
| `ng_board` | `_test` 이미지 ×16 (결함 이미지) | NG |
| `review_board` | `_temp` 8장 + `_test` 8장 (혼합) | NG 또는 REVIEW† |

> † REVIEW 시나리오는 `_test` 이미지에서 모델이 review_band 구간(0.3~0.5) 신뢰도를 출력할 때 재현됩니다. 충분히 학습된 모델에서 정상 동작합니다.

#### UI 흐름

1. 보드 선택 → **검사 시작** 클릭
2. 서버에서 16칸 일괄 추론 후 결과 반환
3. 프론트엔드가 칸별 **순차 스캔 애니메이션** (230ms 간격)
   - 스캔 중: 노란색 하이라이트 + 회전 아이콘
   - 결과: OK=초록 ✓ / NG=빨강 ✕ / REVIEW=주황 ?
4. 진행 카운터 "검사 중 7/16" 표시
5. 보드 최종 판정 배지 + 칸별 집계 (OK/NG/REVIEW 개수)
6. REVIEW 칸 존재 시 **하단 수동 분류 패널** 표시
   - 각 REVIEW 칸의 이미지 + 결함 정보 카드
   - 운영자가 정상/결함 직접 선택 → 전체 선택 후 확인 → 판정 확정

#### 보드 판정 집계

```
칸별 판정을 집계하여 보드 전체 판정:
- NG 칸이 1개라도 있으면  → 보드 NG
- NG 없고 REVIEW 칸 존재 → 보드 REVIEW
- 전체 OK               → 보드 OK
```

### 가상 보드 생성

```bash
python web_hwang/tools/build_demo_boards.py \
  --raw-data dataset/PCBData \
  --group group00041 \
  --rows 4 --cols 4
```

생성 결과: `web_hwang/samples/boards/{ok,ng,review}_board.{jpg,_map.json}`

> ※ 가상 보드는 데모 전용 합성 이미지입니다. DeepPCB 원본에는 2D 위치 정보가 없어 실제 보드 복원이 아닙니다.

---

## 실행 스크립트

### `scripts/run_train.bat` / `run_train.sh` — 대화형 학습 실행

전처리 → 설정 확인 → 학습(신규/이어학습)까지 대화형으로 안내합니다.

```bash
# Windows
scripts\run_train.bat

# Linux/Mac
bash scripts/run_train.sh
```

흐름:
1. 현재 디렉터리 표시
2. 전처리 실행 여부 확인 (Y/N, 기본 N)
3. 현재 config 표시 (`show_config.py`)
4. 학습 진행 여부 확인 (Y/N, 기본 N)
5. 체크포인트 존재 시 이어학습 여부 확인 (Y/N, 기본 N)
6. `python src/train.py [--resume]` 실행

### `scripts/reset_to_main.bat` / `reset_to_main.sh` — main 브랜치 완전 리셋

> ⚠️ **파괴적 동작**: 모든 로컬 변경사항을 삭제하고 `origin/main`으로 완전 리셋합니다.

- `git fetch origin` → `git reset --hard origin/main`
- `preprocessed_data/`, `dataset/`, `runs/`, `weights/` 삭제
- `git clean -fdx`

### `scripts/show_config.py` — 설정 출력

`config.yaml`의 전체 설정을 정렬된 테이블 형태로 출력합니다.

```
+-----------------------------------+---------------------------+
| Parameter                         | Value                     |
+-----------------------------------+---------------------------+
| env                               | local                     |
| train.epochs                      | 100                       |
| judge.conf_threshold              | 0.5                       |
+-----------------------------------+---------------------------+
```

---

## 실행 가이드

### 전체 실행 순서

```bash
# 0. 의존성 설치
pip install -r requirements.txt

# 1. 전처리 (DeepPCB → YOLO 포맷 변환 + 70/20/10 split)
python src/preprocess.py

# 2. 모델 학습 (기본: yolov8n, 100 에폭)
python src/train.py

# 2-1. (선택) K-Fold 교차 검증
python src/train_kfold.py

# 2-2. (선택) 하이퍼파라미터 튜닝 — 매우 오래 걸림
python src/tune.py

# 3. 테스트 세트 평가
python src/evaluate.py

# 4. 단일 이미지 판정
python src/pcb_inspect.py preprocessed_data/images/test/<이미지>.jpg

# 5. 웹 데모 실행
uvicorn web_hwang.app:app --reload --port 8000
```

### 통합 학습 스크립트 (대화형)

```bash
# Windows
scripts\run_train.bat

# Linux/Mac
bash scripts/run_train.sh
```

---

## 스모크 테스트

신규 환경 셋업 후 전체 파이프라인이 정상 동작하는지 빠르게 확인합니다.

```bash
# 1. 의존성 설치
pip install -r requirements.txt

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
python src/pcb_inspect.py preprocessed_data/images/test/<아무_이미지>.jpg
# 기대: 판정(OK/NG/REVIEW) + 결함 목록 출력

# 7. 웹 데모 확인
uvicorn web_hwang.app:app --reload --port 8000
# 기대: http://localhost:8000 접속 후 검사 동작 확인
```

---

## 브랜치 분담

| 브랜치 | 내용 |
|:------|:----|
| `main` | 베이스라인, 리뷰 완료 코드만 merge |
| `feature/field-board` | 전체 보드 격자 순차 검사 웹 데모 |
| `feature/preprocessing` | `src/preprocess.py` 전처리 실험, EDA |
| `feature/model-training` | `src/train.py` 하이퍼파라미터 실험 (n→s→m, lr, aug) |
| `feature/eda-viz` | `src/visualize.py` 클래스 분포·bbox 히스토그램·결과 시각화 |
