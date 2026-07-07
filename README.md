# YOLOv8 기반 PCB 결함 탐지 및 Review Station 시스템
<img width="6400" height="6400" alt="merged_group20085_10x10_03_colored" src="https://github.com/user-attachments/assets/3657b2cc-62bc-49dd-a981-aafe2c73e440" />

YOLOv8 5-Fold 앙상블 모델을 통해 PCB(인쇄회로기판) 표면 결함을 고정밀로 탐지하고, 최종적으로 작업자가 데스크톱 GUI(Review Station)를 통해 빠르고 직관적으로 결함을 검토하고 판정할 수 있도록 돕는 통합 시스템입니다.

> **핵심 차별점** — 단순 bounding box 검출에 그치지 않고, 5개의 모델이 도출한 앙상블 결과를 바탕으로 높은 재현율(Recall)을 달성하여 불량 놓침을 최소화합니다. 또한, 작업자 피로도를 줄이기 위한 단축키 중심의 GUI 리뷰 스테이션을 최종 결과물로 제공합니다.

⚠️ **필수 안내: Git LFS 설치**
이 프로젝트는 대용량 모델 가중치 파일(`.pt`)을 관리하기 위해 Git LFS 전용 저장소를 사용합니다. 저장소를 다운로드(Clone)하거나 최신 버전을 받아오실 때 가중치 파일이 정상적으로 받아지려면 **[Git LFS](https://git-lfs.com/)**가 로컬 PC에 반드시 설치되어 있어야 합니다. (미설치 시 `.pt` 파일이 1KB 미만의 텍스트 포인터로 받아져 실행 시 에러가 발생합니다.)

🚀 **초기 셋업 안내 (필수)**
프로젝트를 처음 다운로드(Clone) 받은 후, 가장 먼저 다음 두 단계를 순서대로 수행하여 실행 환경을 구성해 주세요.
1. **가상환경 세팅 (`install.bat` 또는 `install.sh`)**: 프로젝트 루트 디렉터리에서 실행하여 해당 PC에 맞는 독립적인 파이썬 가상환경(`venv`)을 생성하고 필요한 필수 패키지들을 설치합니다.
2. **데이터 전처리 (`scripts\run_preprocess.bat` 또는 `.sh`)**: 원본 데이터를 모델이 인식할 수 있는 포맷으로 정제합니다. 가상환경 세팅 완료 후 이어서 바로 실행해 줍니다.

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
├── app_back/                # [최종 산출물] 데스크톱 GUI Review Station 앱
│   ├── run.py               # 앱 실행 진입점
│   ├── main_ui.py           # UI 및 이벤트 핸들링
│   ├── vision_viewer.py     # 이미지 뷰어 모듈
│   ├── inference_worker.py  # 앙상블 모델 추론 스레드
│   └── README.md
│
├── scripts/                 # 파이프라인 단계별 실행 보조 스크립트 (.bat, .sh)
│   ├── run_back_app.bat     # 수동 검토용 GUI 앱(app_back) 실행
│   ├── run_front_app.bat    # 실시간 모니터링 GUI 앱(app_front) 실행
│   ├── run_preprocess.bat   # 데이터 전처리 실행
│   ├── run_kfold.bat        # 5-Fold 학습 실행
│   └── README.md
│
├── src/                     # 핵심 머신러닝 파이프라인 소스 코드
│   ├── preprocess.py        # DeepPCB 포맷 → YOLO 변환
│   ├── train_kfold.py       # 5-Fold 교차 검증 및 모델 학습
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
프로젝트 내의 모든 모듈(`src` 및 `app_back`)은 `config.yaml`의 설정을 기반으로 동작합니다. 경로 지정, 데이터 분할 비율, 학습 에폭(Epoch), 평가 임계값(Threshold) 등 시스템 전반의 핵심 파라미터를 여기서 일괄 관리합니다.

### 데이터 파이프라인 및 학습: `src/` 디렉터리
- **전처리 (`preprocess.py`):** DeepPCB의 고유 포맷을 읽어와 YOLO 모델이 인식할 수 있는 정규화된 xywh 텍스트 라벨 형식으로 변환합니다. 변환 결과는 `preprocessed_data/`에 생성됩니다.
- **K-Fold 모델 학습 (`train_kfold.py`):** 원본 데이터가 특정 클래스에 치우치지 않도록 분할(Stratified K-Fold)하여 총 5개의 모델을 학습시킵니다. 학습의 결과로 도출된 `best.pt` 가중치들은 `weights/` 폴더에 모이게 됩니다.
- **평가 (`generate_results.py`):** `web_test` 폴더 내에서 5개 모델의 앙상블을 통해 재현율(Recall)과 mAP 수치를 산출합니다.

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
   scripts\run_front_app.bat
   ```
   또는 수동 검토용 앱 실행:
   ```cmd
   scripts\run_back_app.bat
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

> 앱의 세부 화면 구성, 브러쉬 사용법 및 단축키 안내는 `app_back/README.md`를 참고하시기 바랍니다.

---

## 6. GPU 서버 모델 학습(Tune) 핵심 가이드

GPU 서버에서 모델 튜닝 및 학습을 진행할 때 필수적인 핵심 명령어 모음입니다. 상세 가이드는 `GPU_SERVER_USER_GUIDE.md`를 참고하세요.

1. **GPU 서버 SSH 접속 및 작업 폴더 이동**
   로컬 PC의 터미널(PowerShell 또는 Mac 터미널)을 열고 서버에 접속합니다.
   ```bash
   ssh team03@***.***.***.***
   ```
   비밀번호를 입력하여 접속한 뒤, 프로젝트 폴더로 이동하고 가상환경을 켭니다.
   ```bash
   cd ~/workspace/pcb-defect-inspection
   source ~/.bashrc
   conda activate ai
   pip install -r requirements.txt  # 최초 1회만 실행
   ```

2. **Git LFS 원본 데이터 풀링** (dataset.zip 등 대용량 파일 받아오기)
   ```bash
   # Git LFS 패키지 설치 (명령어 에러 방지용 필수)
   conda install -c conda-forge git-lfs -y
   
   git lfs install
   git lfs pull
   ```

3. **GitHub 최신 변경사항 서버에 동기화 (업데이트)**
   로컬 PC에서 수정한 코드를 GPU 서버에 100% 동일하게 반영(덮어쓰기)하려면 다음 명령어를 실행합니다.
   ```bash
   git fetch origin
   git reset --hard origin/main
   ```

4. **프로젝트 완전 초기화 및 재설치 (오류 발생 시 최후의 수단)**
   설정이 심하게 꼬였거나 폴더를 실수로 삭제한 경우, 프로젝트를 통째로 지우고 처음부터 다시 세팅하는 방법입니다.
   ```bash
   # 1. 작업 폴더로 이동하여 기존 프로젝트 통째로 삭제
   cd ~/workspace
   rm -rf pcb-defect-inspection
   
   # 2. 프로젝트 다시 다운로드 및 폴더 진입
   git clone https://github.com/june0922/pcb-defect-inspection.git
   cd pcb-defect-inspection
   
   # 3. 가상환경 활성화 및 Git LFS 패키지 설치 (명령어 에러 방지)
   source ~/.bashrc
   conda activate ai
   conda install -c conda-forge git-lfs -y
   
   # 4. 대용량 데이터셋(LFS) 받아오기 및 데이터 전처리 다시 수행
   git lfs install
   git lfs pull
   python src/preprocess.py
   ```

5. **기존 학습 결과만 초기화 (새로운 학습 전 권장)**
   새로운 학습을 시작하기 전에 이전에 생성된 학습 결과와 로그만 깨끗하게 지우려면 다음 명령어를 실행합니다. (필요한 결과는 미리 로컬로 다운로드하세요)
   ```bash
   rm -rf runs weights
   ```

6. **학습/튜닝 스크립트 실행 (GPU 번호 지정 필수)**
   목적에 맞는 스크립트를 선택하여 실행합니다. 타 사용자와 겹치지 않게 사용할 GPU 번호(`CUDA_VISIBLE_DEVICES=0` 등)를 반드시 지정해 주세요.
   
   ```bash
   # [옵션 A] 하이퍼파라미터 튜닝 실행
   CUDA_VISIBLE_DEVICES=0 bash scripts/run_tune.sh

   # [옵션 B] 일반 단일 모델 학습 실행
   CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh

   # [옵션 C] 5-Fold 교차 검증 앙상블 학습 실행 (시간이 가장 오래 걸립니다)
   CUDA_VISIBLE_DEVICES=0 bash scripts/run_kfold.sh
   ```
   *(스크립트 실행 시 데이터 전처리 수행 여부(Y/N) 및 학습 진행 여부를 묻는 프롬프트가 나타납니다.)*

> **💡 참고 (학습 자동 저장 로직)**
> 학습이나 튜닝 스크립트는 중간에 멈추거나 완료되었을 때 제일 성능이 좋은 세팅(`best.pt` 또는 `best_hyperparameters.yaml`)을 `weights/`와 `runs/` 디렉토리에 자동으로 안전하게 백업 및 저장하고 종료됩니다. 켜두고 다른 작업을 하셔도 무방합니다.

7. **터미널(SSH) 연결 종료 후에도 학습 유지하기 (tmux 활용 - 권장)**
   로컬 PC의 터미널 창을 끄면 서버의 스크립트도 강제 종료됩니다. 이를 방지하고 컴퓨터를 끄더라도 학습이 계속 돌아가게 하려면 `tmux`를 사용하세요.
   ```bash
   # 1. 새로운 가상 터미널 열기
   tmux new -s train_session
   
   # 💡 만약 "duplicate session" 에러가 뜨면 기존 세션이 이미 있는 것입니다.
   # 기존 방에 그냥 들어가려면: tmux attach -t train_session
   # 기존 방을 삭제하고 아예 새로 파려면: tmux kill-session -t train_session 입력 후 다시 tmux new -s train_session 실행
   # 단독으로 특정 세션만 삭제하려면: tmux kill-session -t 세션이름 (예: tmux kill-session -t train_session)
   
   # 2. 가상환경 활성화 및 위 6번의 스크립트 실행 (예: 튜닝)
   conda activate ai
   CUDA_VISIBLE_DEVICES=0 bash scripts/run_tune.sh
   
   # 3. 백그라운드로 빠져나오기 (제일 중요 ⭐️)
   # 키보드에서 `Ctrl + b` 를 한 번 눌렀다 뗀 후, 영어 알파벳 `d` 를 누릅니다.
   # 이제 로컬 PC의 터미널을 안전하게 끄셔도 됩니다!
   
   # 4. 나중에 진행 상황 다시 확인하기 (서버 재접속 후)
   tmux attach -t train_session
   ```

8. **학습 결과물 내 로컬 PC로 다운로드 (SCP 활용)**
   ⚠️ 서버가 아닌 **내 로컬 PC(윈도우)의 새로운 PowerShell/CMD 창**에서 실행하세요.
   ```cmd
   # runs 폴더(학습 로그 및 세부 결과) 전체 다운로드
   scp -r team03@***.***.***.***:~/workspace/pcb-defect-inspection/runs C:\Users\compu\Downloads

   # weights 폴더(최종 산출된 가중치 파일들) 전체 다운로드
   scp -r team03@***.***.***.***:~/workspace/pcb-defect-inspection/weights C:\Users\compu\Downloads
   ```
