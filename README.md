# YOLO26 Small 5-Fold 앙상블 기반 PCB 결함 탐지 시스템

## app_front — 실시간 자동 검사 모니터링
<img width="800" height="418" alt="front_-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/05cf78d6-8f84-418c-9151-24e3daa3f969" />

## app_back — REVIEW 타일 수동 판정 리뷰 스테이션
<img width="800" height="418" alt="back_-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/6af7445e-c551-4a84-a7cb-a975bcef5a85" />

## 소개
YOLO26 Small(yolo26s) 5-Fold 앙상블 모델로 PCB(인쇄회로기판) 표면 결함을 고정밀 탐지하는 통합 시스템입니다.  
`app_front`가 PCB 이미지를 640×640 타일로 분할하여 자동 검사(PASS/FAIL/REVIEW)하고, 결과를 SQLite DB에 실시간 저장합니다.  
`app_back`은 DB에서 REVIEW 타일을 수신하여 작업자가 키보드 단축키로 빠르게 최종 판정을 내릴 수 있도록 합니다.

> **핵심 차별점** — 단순 bounding box 검출에 그치지 않고, 5개 모델의 WBF(Weighted Boxes Fusion) 앙상블 결과를 바탕으로 높은 재현율(Recall)을 달성하여 불량 놓침을 최소화합니다. 클래스별 독립 REVIEW 밴드(min/max %)로 유연한 감도 조절이 가능합니다.

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
5. [데스크톱 GUI 앱 구성](#5-데스크톱-gui-앱-구성)
6. [GPU 서버 모델 학습 이용 가이드](#6-gpu-서버-모델-학습-이용-가이드)

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
├── app_front/               # [최종 산출물] 실시간 자동 PCB 결함 검사 모니터링
│   ├── run.py               # 앱 실행 진입점
│   ├── run_app.bat          # Windows: app_front 단독 실행
│   ├── run_app.sh           # macOS/Linux: app_front 단독 실행
│   ├── main_ui.py           # 메인 윈도우 (MainWindow, SettingsDialog)
│   ├── inspection_worker.py # 타일 검사 워커 스레드
│   ├── global_view.py       # PCB 전체 뷰어 + 그리드 오버레이
│   ├── vision_viewer.py     # 타일 확대 뷰어 (읽기전용)
│   └── README.md
│
├── app_back/                # [최종 산출물] REVIEW 결함 수동 판정 리뷰 스테이션
│   ├── run.py               # 앱 실행 진입점
│   ├── run_app.bat          # Windows: app_back 단독 실행
│   ├── run_app.sh           # macOS/Linux: app_back 단독 실행
│   ├── main_ui.py           # 메인 윈도우 (DB 폴링, 판정 단축키, bbox 편집 연동)
│   ├── vision_viewer.py     # 결함 오버레이 대화형 뷰어 (bbox 드래그 이동/리사이즈)
│   └── README.md
│
├── db/                      # SQLite 데이터베이스 (SSOT)
│   ├── database.py          # DB 헬퍼 함수 (init, upsert, fetch, settings)
│   ├── inspection.db        # SQLite DB 파일 (자동 생성, WAL 모드)
│   └── README.md
│
├── scripts/                 # 전처리·학습·초기화 실행 스크립트 (.bat / .sh)
│   ├── run_preprocess.bat/sh
│   ├── run_train.bat/sh
│   ├── run_kfold.bat/sh
│   ├── run_tune.bat/sh
│   ├── run_train_tune.bat/sh
│   ├── reset_to_main.bat/sh
│   └── README.md
│
├── src/                     # 핵심 머신러닝 파이프라인 소스 코드
│   ├── preprocess.py        # DeepPCB 포맷 → YOLO 변환 + 그룹 단위 분할
│   ├── merge_images.py      # 서브 이미지 합성 (데이터 확장)
│   ├── train.py             # 단일 모델 학습
│   ├── train_kfold.py       # 5-Fold K-Fold 앙상블 학습
│   ├── tune.py              # 하이퍼파라미터 탐색
│   ├── train_tune.py        # 튜닝 결과 적용 정밀 학습
│   ├── utils.py             # config 로드 + 환경별 경로 분기
│   └── README.md
│
├── weights/                 # 베이스 모델 및 앙상블 최종 가중치
│   ├── yolo26n.pt           # YOLO26 Nano 베이스 (~5.4 MB)
│   ├── yolo26s.pt           # YOLO26 Small 베이스 (~19.5 MB, 실제 사용)
│   ├── best_fold_1~5.pt     # 5-Fold K-Fold 최적 가중치 (~6 MB × 5)
│   └── README.md
│
├── notebooks/               # Google Colab 학습·튜닝 노트북
│   ├── pcb_train_colab.ipynb
│   ├── pcb_kfold_colab.ipynb
│   ├── pcb_tune_colab.ipynb
│   ├── pcb_train_tune_colab.ipynb
│   └── README.md
│
├── dataset/                 # 원본 DeepPCB 데이터셋 폴더
├── preprocessed_data/       # 전처리 완료 YOLO 형식 데이터 (자동 생성)
└── runs/                    # 모델 학습 로그 및 산출물 (자동 생성)
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
프로젝트 내의 모든 모듈(`src` 폴더)은 `config.yaml`의 설정을 기반으로 동작합니다. 경로 지정, 데이터 분할 비율, 학습 에폭(Epoch) 등 시스템 전반의 핵심 파라미터를 여기서 일괄 관리합니다.

### 데이터 파이프라인 및 학습: `src/` 디렉터리
- **전처리 (`preprocess.py`):** DeepPCB 포맷을 YOLO 포맷으로 변환합니다. DeepPCB의 11개 그룹(동일 회로 설계)을 기준으로 greedy 분할하여 train/val/test 간 데이터 누수를 방지합니다.
- **K-Fold 모델 학습 (`train_kfold.py`):** `StratifiedGroupKFold`로 그룹 경계를 유지하며 5개 모델을 학습합니다. 결과 가중치는 `weights/best_fold_1~5.pt`로 저장됩니다.
- **하이퍼파라미터 튜닝 (`tune.py`):** 최적 학습 파라미터를 탐색합니다.

> 각 모듈의 세부 설명은 `src/README.md`를 참고하세요.

### 단일 진실 공급원: `db/` 디렉터리
- `app_front`(쓰기)와 `app_back`(읽기)이 SQLite 파일 하나를 공유합니다.
- WAL(Write-Ahead Logging) 모드로 동시 접근을 지원합니다.
- 클래스별 REVIEW 밴드 설정도 DB의 settings 테이블에서 관리합니다.

---

## 4. 실행 가이드

(아래는 Windows 환경 기준 예시입니다. macOS/Linux는 `.bat` → `.sh`로 변경하세요.)

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
4. **앱 실행 (app_front와 app_back을 각자 실행)**
   두 앱은 완전히 독립적인 프로그램입니다 — 어느 쪽을 먼저 켜도, 하나만 켜도 정상 동작합니다.
   ```bash
   # Inspection Monitor (자동 검사)
   app_front\run_app.bat

   # Review Station (수동 판정) — 별도 터미널/창에서
   app_back\run_app.bat
   ```
   - app_back은 시작 즉시 DB 폴링을 시작합니다(더 이상 자체 모델을 로딩하지 않음).
   - app_front에서 `File > Open Folder...`로 검사 폴더를 선택하면 검사가 시작됩니다.

---

## 5. 데스크톱 GUI 앱 구성

본 프로젝트의 **최종 산출물**인 두 GUI 앱이 SQLite DB를 통해 실시간으로 연동됩니다.

### app_front — Inspection Monitor (자동 검사)

PCB 이미지를 640×640 타일로 분할하여 5개 YOLO 모델(yolo26s 기반)의 WBF 앙상블 추론으로 결함을 자동 탐지합니다.

| 판정 | 기준 | 처리 |
|------|------|------|
| PASS | 모든 결함 클래스 신뢰도 < review_min | 정상 — DB 기록 |
| REVIEW | review_min ≤ 신뢰도 ≤ review_max | app_back에 전달 — DB 기록 |
| FAIL | 신뢰도 > review_max (어느 클래스라도) | 불량 확정 — DB 기록 + 경고음 |

- **서펜타인(ㄹ자) 스캔** — 인접 PCB 간 연속성 유지를 위한 타일 순회 순서
- **클래스별 독립 REVIEW 밴드** — `Option > Settings...`에서 각 결함 유형마다 min/max 별도 설정
- **GlobalView** — 컬러 PCB 전체 + 검사 진행 그리드 오버레이 (실시간)
- **Statistics Panel** — Inspected, PASS/FAIL/REVIEW(%), FPY, Throughput, Defect 분포

> 세부 단축키 및 파이프라인 설명은 `app_front/README.md`를 참고하세요.

### app_back — Review Station (수동 판정)

DB에서 3초마다 REVIEW 결함을 폴링하여 필름스트립에 수신합니다. 자체 재추론은 하지 않고
app_front가 저장 시점에 계산한 bbox/클래스/신뢰도를 그대로 읽어 표시합니다.

| 단축키 | 동작 |
|--------|------|
| `Space` | Pass (정상/오탐) |
| `1`~`6` | Fail (결함 클래스 번호) |
| `←` / `→` | 이전 / 다음 결함 이동 |
| `W/A/S/D` | 뷰어 패닝 |
| `Q` / `E` | 뷰어 축소 / 확대 |
| `Shift` | 선택 오버레이 외 숨김 |
| 좌클릭 드래그 | 강조된 결함의 bbox 이동/리사이즈 (DB에 즉시 저장) |

> 세부 단축키 및 DB 폴링 메커니즘은 `app_back/README.md`를 참고하세요.

---

## 6. GPU 서버 모델 학습 이용 가이드

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
   conda install -c conda-forge git-lfs -y  # 최초 1회만 실행
   
   git lfs install  # 최초 1회만 실행
   git lfs pull  # 최초 1회만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   ```

3. **GitHub 최신 변경사항 서버에 동기화 (업데이트)**
   로컬 PC에서 수정한 코드를 GPU 서버에 100% 동일하게 반영(덮어쓰기)하려면 다음 명령어를 실행합니다.
   ```bash
   git fetch origin  # 깃허브 최신 반영 원할 시에만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   git reset --hard origin/main    # 깃허브 최신 반영 원할 시에만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   ```

4. **프로젝트 완전 초기화 및 재설치 (오류 발생 시 최후의 수단)**
   설정이 심하게 꼬였거나 폴더를 실수로 삭제한 경우, 프로젝트를 통째로 지우고 처음부터 다시 세팅하는 방법입니다.
   ```bash
   # 1. 작업 폴더로 이동하여 기존 프로젝트 통째로 삭제
   cd ~/workspace
   rm -rf pcb-defect-inspection  # 파일 꼬였을 시에만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   
   # 2. 프로젝트 다시 다운로드 및 폴더 진입
   git clone https://github.com/june0922/pcb-defect-inspection.git  # 파일 꼬였을 시에만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   cd pcb-defect-inspection
   
   # 3. 가상환경 활성화 및 Git LFS 패키지 설치 (명령어 에러 방지)
   source ~/.bashrc
   conda activate ai
   conda install -c conda-forge git-lfs -y  # 최초 1회만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   
   # 4. 대용량 데이터셋(LFS) 받아오기 및 데이터 전처리 다시 수행
   git lfs install  # 최초 1회만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   git lfs pull  # 최초 1회만 실행 (모델 학습 결과 다운로드 전에 절대 금지)
   python src/preprocess.py
   ```

5. **기존 학습 결과만 초기화 (새로운 학습 전 권장)**
   새로운 학습을 시작하기 전에 이전에 생성된 학습 결과와 로그만 깨끗하게 지우려면 다음 명령어를 실행합니다. (필요한 결과는 미리 로컬로 다운로드하세요)
   ```bash
   rm -rf runs weights  # 모델 학습 결과 다운로드 전에 절대 금지
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
   # 세션 삭제하려면: tmux kill-session -t 세션이름 (예: tmux kill-session -t train_session)
   
   # 2. 가상환경 활성화 및 위 6번의 스크립트 실행 (예: 튜닝)
   cd ~/workspace/pcb-defect-inspection
   source ~/.bashrc
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
