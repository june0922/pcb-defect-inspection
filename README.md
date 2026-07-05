# YOLOv8 기반 PCB 결함 탐지 및 Review Station 시스템

YOLOv8 5-Fold 앙상블 모델을 통해 PCB(인쇄회로기판) 표면 결함을 고정밀로 탐지하고, 최종적으로 작업자가 데스크톱 GUI(Review Station)를 통해 빠르고 직관적으로 결함을 검토하고 판정할 수 있도록 돕는 통합 시스템입니다.

> **핵심 차별점** — 단순 bounding box 검출에 그치지 않고, 5개의 모델이 도출한 앙상블 결과를 바탕으로 높은 재현율(Recall)을 달성하여 불량 놓침을 최소화합니다. 또한, 작업자 피로도를 줄이기 위한 단축키 중심의 GUI 리뷰 스테이션을 최종 결과물로 제공합니다.

⚠️ **필수 안내: Git LFS 설치**
이 프로젝트는 대용량 모델 가중치 파일(`.pt`)을 관리하기 위해 Git LFS 전용 저장소를 사용합니다. 저장소를 다운로드(Clone)하거나 최신 버전을 받아오실 때 가중치 파일이 정상적으로 받아지려면 **[Git LFS](https://git-lfs.com/)**가 로컬 PC에 반드시 설치되어 있어야 합니다. (미설치 시 `.pt` 파일이 1KB 미만의 텍스트 포인터로 받아져 실행 시 에러가 발생합니다.)

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [데이터셋 (DeepPCB)](#2-데이터셋-deeppcb)
3. [설정 파일 및 핵심 소스 코드](#3-설정-파일-및-핵심-소스-코드)
4. [실행 가이드](#4-실행-가이드)
5. [데스크톱 GUI 앱 (Review Station)](#5-데스크톱-gui-앱-review-station)

---

## 1. 프로젝트 구조

```text
ati3_project/
├── config.yaml              # 전체 시스템 설정 (환경, 하이퍼파라미터 등)
├── data.yaml                # YOLO 데이터셋 정의
├── requirements.txt         # Python 의존성
├── install.bat / .sh        # 가상환경 생성 및 의존성 자동 설치 스크립트
├── .gitignore               # 추적 제외 설정
├── .gitattributes           # Git LFS 추적 설정 (*.pt, *.zip)
│
├── app/                     # [최종 산출물] 데스크톱 GUI Review Station 앱
│   ├── run.py               # 앱 실행 진입점
│   ├── main_ui.py           # UI 및 이벤트 핸들링
│   ├── vision_viewer.py     # 이미지 뷰어 모듈
│   ├── inference_worker.py  # 앙상블 모델 추론 스레드
│   └── README.md
│
├── scripts/                 # 파이프라인 단계별 실행 보조 스크립트 (.bat, .sh)
│   ├── run_app.bat          # GUI 앱 실행
│   ├── run_preprocess.bat   # 데이터 전처리 실행
│   ├── run_kfold.bat        # 5-Fold 학습 실행
│   └── README.md
│
├── src/                     # 핵심 머신러닝 파이프라인 소스 코드
│   ├── preprocess.py        # DeepPCB 포맷 → YOLO 변환
│   ├── train_kfold.py       # 5-Fold 교차 검증 및 모델 학습
│   ├── evaluate.py          # 학습 모델 성능 평가
│   └── README.md
│
├── weights/                 # 사전 학습 및 파인튜닝된 앙상블 가중치(best_fold_1~5.pt)
│   └── README.md
│
├── notebooks/               # Colab 환경용 튜닝 및 학습 노트북 파일 모음
│   └── README.md
│
├── dataset/                 # 원본 DeepPCB 데이터셋 폴더
├── preprocessed_data/       # 전처리가 완료된 YOLO 형식 데이터 폴더 (자동 생성)
└── runs/                    # 모델 학습 로그 및 산출물 보관 (자동 생성)
```

---

## 2. 데이터셋 (DeepPCB)

[DeepPCB](https://github.com/tangsanli5201/DeepPCB)는 PCB 결함 검출 연구를 위한 공개 데이터셋입니다. 640×640 해상도의 이미지에 대해 정상 템플릿과 실제 결함 이미지가 짝을 이루어 제공됩니다.

### 결함 클래스 (6종)

DeepPCB 원본 type 1-6을 YOLO cls 0-5로 매핑하여 학습에 사용합니다.

| YOLO cls | 클래스명     | 설명                               |
|:--------:|:----------:|:----------------------------------|
| 0        | **open**     | 단선 — 회로 배선이 끊어진 결함            |
| 1        | **short**    | 단락 — 인접 배선이 연결된 결함            |
| 2        | **mousebite**| 마우스바이트 — 배선 가장자리가 불규칙하게 깎인 결함 |
| 3        | **spur**     | 돌기 — 배선에서 불필요한 돌출이 발생한 결함     |
| 4        | **copper**   | 동잔류 — 제거되어야 할 구리가 남아있는 결함     |
| 5        | **pinhole**  | 핀홀 — 배선에 미세한 구멍이 생긴 결함        |

---

## 3. 설정 파일 및 핵심 소스 코드

### 중앙 설정 관리: `config.yaml`
프로젝트 내의 모든 모듈(`src` 및 `app`)은 `config.yaml`의 설정을 기반으로 동작합니다. 경로 지정, 데이터 분할 비율, 학습 에폭(Epoch), 평가 임계값(Threshold) 등 시스템 전반의 핵심 파라미터를 여기서 일괄 관리합니다.

### 데이터 파이프라인 및 학습: `src/` 디렉터리
- **전처리 (`preprocess.py`):** DeepPCB의 고유 포맷을 읽어와 YOLO 모델이 인식할 수 있는 정규화된 xywh 텍스트 라벨 형식으로 변환합니다. 변환 결과는 `preprocessed_data/`에 생성됩니다.
- **K-Fold 모델 학습 (`train_kfold.py`):** 원본 데이터가 특정 클래스에 치우치지 않도록 분할(Stratified K-Fold)하여 총 5개의 모델을 학습시킵니다. 학습의 결과로 도출된 `best.pt` 가중치들은 `weights/` 폴더에 모이게 됩니다.
- **평가 (`evaluate.py`):** 모델이 불량을 놓치지 않고 얼마나 잘 찾아내는지를 판단하기 위해 재현율(Recall)과 mAP 수치를 상세 분석합니다.

> 각 스크립트별 세부 설명은 `src/README.md`에서 확인하실 수 있습니다.

---

## 4. 실행 가이드

모든 스크립트는 `scripts/` 폴더에 배치(.bat) 및 쉘(.sh) 확장자로 제공되므로 사용자의 OS 환경에 맞춰 간편하게 파이프라인을 구동할 수 있습니다. (아래는 윈도우 환경 기준 예시입니다.)

1. **의존성 설치 (가상환경 셋업)**
   최초 1회만 수행합니다. 자동으로 `venv`를 만들고 필요한 라이브러리를 설치합니다.
   ```bash
   install.bat
   ```
2. **데이터 전처리**
   압축 해제된 `dataset`을 바탕으로 YOLO 포맷 변환을 수행합니다.
   ```bash
   scripts\run_preprocess.bat
   ```
3. **K-Fold 앙상블 학습**
   5-Fold 교차 검증을 통해 5개의 가중치(`weights/best_fold_1~5.pt`)를 생성합니다. (GPU 환경 권장)
   ```bash
   scripts\run_kfold.bat
   ```
4. **리뷰 스테이션 실행**
   최종 산출물인 데스크톱 GUI 애플리케이션을 구동하여 이미지 검사를 진행합니다.
   ```bash
   scripts\run_app.bat
   ```

---

## 5. 데스크톱 GUI 앱 (Review Station)

본 프로젝트의 **최종 산출물**로, 학습된 AI 모델이 1차적으로 탐지해 낸 결함 후보들을 작업자가 최종적으로 검토하고 판정하기 위한 전용 애플리케이션입니다. 

`weights/` 폴더 내의 **YOLOv8 Nano 5-Fold 앙상블 가중치**를 모두 로드한 뒤, 백그라운드 스레드에서 WBF(Weighted Boxes Fusion) 앙상블을 수행하여 결함 예측의 재현율과 신뢰도를 대폭 높였습니다.

### 직관적인 판정 UX 및 대화형 뷰어
작업자의 반복적인 마우스 조작 피로도를 없애기 위해 강력한 대화형 뷰어와 키보드 전역 단축키를 지원합니다.
* **듀얼 뷰 동기화:** 전체 화면(Global View)과 확대 화면(Local View)이 실시간으로 동기화되어 결함 위치를 한눈에 파악
* **초고속 판정:** 키보드 **`1`~`6`** (유형별 Fail) 또는 **`Space`** (Pass - 먼지/정상)를 눌러 즉각 판정 후 다음으로 자동 이동
* **뷰어 제어:** 방향키(`←`, `→`)로 결함 후보 순회 및 `W/A/S/D` (패닝), `Q/E` (줌) 전역 단축키 지원
* **브러쉬 마스킹:** 좌클릭 드래그로 이미지의 오탐 부위를 지우거나 가리는 실시간 편집 기능 및 `F5` 수동 재연산 기능
* 최종 판정 결과는 JSON 파일로 저장(`Ctrl+S`)되어 후속 통계나 재학습 데이터로 즉시 활용 가능

> 앱의 세부 화면 구성, 브러쉬 사용법 및 단축키 안내는 `app/README.md`를 참고하시기 바랍니다.
