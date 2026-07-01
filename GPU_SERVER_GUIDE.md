# GPU 서버 사용 및 3팀(team03) 프로젝트 배포 가이드

본 문서는 제공된 GPU 서버 사용 안내서 원문과, **3팀(team03)**의 `pcb-defect-inspection` 프로젝트를 서버 환경에 맞추어 구성하고 실행하는 구체적인 방법을 정리한 문서입니다.

---

## 1. 3팀(team03) 프로젝트 배포 및 실행 가이드

우리 프로젝트(`pcb-defect-inspection`)는 **상대 경로(`env: local`)**를 기준으로 모든 파이프라인이 자동화되어 있어, 서버 내에서 파일 경로를 직접 수정하지 않고 즉시 구동할 수 있습니다.

### Step 1. GPU 서버 SSH 접속
터미널(PowerShell 또는 CMD 등)을 열고 3팀 계정으로 접속합니다.
```bash
ssh team03@165.246.170.53
```
* 최초 로그인 시 비밀번호 변경을 요구받을 수 있으니 안내에 따라 변경합니다.

### Step 2. 작업 공간으로 이동 및 프로젝트 Clone
3팀의 작업 공간(`~/workspace`)으로 이동한 뒤, 깃허브 저장소를 복사합니다.
```bash
# workspace 이동 (실제 경로: /data/inha/team03/workspace)
cd ~/workspace

# 저장소 클론 및 디렉토리 진입
git clone https://github.com/june0922/pcb-defect-inspection.git
cd pcb-defect-inspection
```

### Step 3. 대용량 데이터셋(LFS) 다운로드 확인
대용량 파일인 `dataset.zip`이 Git LFS를 통해 완전하게 내려받아졌는지 검증합니다. (파일 크기가 수십 KB 수준의 포인터 파일로만 복사되어 있다면 아래 명령어를 실행해야 합니다.)
```bash
git lfs pull
```

### Step 4. Conda 환경 활성화 및 의존성 설치
서버에 사전 설치된 `ai` 가상환경을 활성화하고 추가 패키지를 검증합니다.
```bash
# 1. Conda 초기화 및 활성화
source ~/.bashrc
conda activate ai

# 프롬프트가 (ai) team03@... 으로 변경되는지 확인

# 2. 추가 필요한 패키지 설치
pip install -r requirements.txt
```

### Step 5. 설정 파일 (`config.yaml`) 확인
기본 설정 파일(`config.yaml`)의 `env: local` 옵션은 상대 경로(`project_root: .`, `raw_data: dataset/PCBData`)로 세팅되어 있어 **서버에서도 그대로 사용 가능**합니다.
* 만약 실제 학습 횟수(Epochs)를 설정하고 싶다면, 서버에서 파일을 수정합니다.
```bash
# 텍스트 에디터로 설정 수정 (예: train.epochs 값을 100으로 설정)
nano config.yaml
```

### Step 6. 데이터 전처리 (압축 해제 및 YOLO 변환)
LFS로 가져온 `dataset.zip`을 해제하고 YOLO 포맷으로 변환해 주는 전처리 파이프라인을 구동합니다.
```bash
python src/preprocess.py
```
* 동작 완료 후, `data/processed/` 폴더 내에 `train/val/test` 데이터 파티셔닝이 완료됩니다.

### Step 7. GPU 할당 확인 및 학습(Training) 실행
3팀의 GPU 할당 일정을 준수하며 훈련을 시작합니다.
```bash
# 1. 현재 GPU 사용 현황 확인
nvidia-smi

# 2. GPU 0번(또는 1번)을 지정하여 YOLO 학습 실행
CUDA_VISIBLE_DEVICES=0 python src/train.py
```
* 학습 결과 모델(`best.pt`)은 `weights/best.pt` 경로에 자동으로 저장됩니다.

### Step 8. Jupyter Lab 원격 포트 포워딩 연결 (선택 사항)
서버의 웹 기반 주피터 환경을 사용하려면 **3팀 권장 포트인 `8883`**을 활용합니다.

1. **서버 터미널**에서 실행:
   ```bash
   jupyter lab --port 8883 --no-browser
   ```
2. **로컬 PC 터미널**을 새로 열어 포트 포워딩 명령 실행:
   ```bash
   ssh -L 8883:localhost:8883 team03@165.246.170.53
   ```
3. 로컬 PC 브라우저 주소창에 `http://localhost:8883`을 입력하여 접속합니다.

---

## 2. AI 실습 서버 접속 및 사용 안내 (원문)

본 서버는 팀별 AI 학습 및 실습을 위한 Ubuntu 기반 실습 서버입니다. 각 팀은 독립된 계정과 Conda 기반 Python 환경을 사용하며, 실습 파일은 팀별 전용 작업공간에 저장합니다.

### 1. 접속 정보
* **서버 주소:** 165.246.170.53
* **접속 방식:** SSH 접속
* **운영체제:** Ubuntu Server
* **실습 환경:** Conda ai 환경
* **기본 작업 폴더:** `~/workspace`
* **공용 데이터/모델 폴더:** `/data/inha/shared`

#### 팀별 계정 및 포트
* **1팀:** team01 (홈: `/data/inha/team01`, Jupyter 포트: `8881`)
* **2팀:** team02 (홈: `/data/inha/team02`, Jupyter 포트: `8882`)
* **3팀:** team03 (홈: `/data/inha/team03`, Jupyter 포트: `8883`)
* **4팀:** team04 (홈: `/data/inha/team04`, Jupyter 포트: `8884`)
* **5팀:** team05 (홈: `/data/inha/team05`, Jupyter 포트: `8885`)
* *초기 비밀번호는 별도 공지. 최초 로그인 시 비밀번호 변경을 요구받을 수 있습니다.*

### 2. 서버 접속 방법
* 3팀 접속 예시: `ssh team03@165.246.170.53`

### 3. 디렉토리 구조
```text
/data/inha
├── team01 ~ team05
│   ├── workspace (팀별 작업 공간)
│   └── miniforge3 (Conda 설치 경로)
└── shared
    ├── datasets (공용 데이터셋)
    ├── models (공용 모델 파일)
    └── examples (공용 예제 코드)
```

### 4. Conda 환경 사용 방법
서버 접속 후 아래 명령을 실행하여 실습용 가상환경 `ai`를 활성화합니다.
```bash
source ~/.bashrc
conda activate ai
```
* 활성화 성공 시 프롬프트에 `(ai)` 표시가 붙습니다. (예: `(ai) team03@JAVIS:~/workspace$`)
* 종료 시에는 `conda deactivate`를 입력합니다.

### 5. 사전 설치된 주요 Python 모듈
* **기본 연산/시각화:** numpy, pandas, matplotlib, scikit-learn, opencv-python
* **딥러닝/LLM:** torch, torchvision, torchaudio, transformers, datasets, accelerate, sentencepiece
* **노트북 환경:** jupyterlab, notebook

### 6. 추가 모듈 설치 방법
패키지는 반드시 `conda activate ai`로 가상환경을 활성화한 다음 설치해야 합니다.
```bash
conda activate ai
pip install 패키지명
```

### 7. GPU 사용 확인 및 할당
GPU 상태는 `nvidia-smi` 명령으로 상시 확인하고, 본인 팀에 할당된 일정을 준수하여 특정 GPU를 지정해서 프로세스를 실행합니다.
* **GPU 0번 지정 실행:** `CUDA_VISIBLE_DEVICES=0 python train.py`
* **GPU 1번 지정 실행:** `CUDA_VISIBLE_DEVICES=1 python train.py`

### 8. 사용 규칙
1. 본인 팀 계정(`team03`)만 사용합니다.
2. 작업 파일은 반드시 `~/workspace` 아래에 저장합니다.
3. 공용 데이터셋과 모델은 `/data/inha/shared`를 먼저 참조해 공간을 절약합니다.
4. 대용량 모델, 체크포인트, 캐시 파일은 실습 종료 후 정리하여 디스크 공간을 확보합니다.
5. GPU 사용 전 `nvidia-smi`로 다른 프로세스의 사용 상태를 체크합니다.
6. 다른 팀의 프로세스를 임의로 종료하지 않습니다.
7. 중요한 결과물은 개인 PC나 GitHub에 별도 백업합니다.
