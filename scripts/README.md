# Scripts Directory

본 `scripts` 폴더는 프로젝트 내의 다양한 작업들(전처리, 학습, 하이퍼파라미터 튜닝, GUI 프로그램 실행 등)을 간편하게 실행할 수 있도록 작성된 쉘 스크립트(`.sh`) 및 배치 파일(`.bat`)을 보관하고 있습니다.

## 주요 스크립트 안내

각 스크립트는 **프로젝트 루트 경로**에서 실행하는 것을 원칙으로 설계되어 있습니다.

* **`run_preprocess.bat` / `run_preprocess.sh`**
  * 원본 데이터셋의 전처리 및 학습을 위한 데이터 분할(`src/preprocess.py`)을 실행합니다.
* **`run_train.bat` / `run_train.sh`**
  * 단일 모델 기반의 학습 파이프라인(`src/train.py`)을 실행합니다.
* **`run_kfold.bat` / `run_kfold.sh`**
  * 5-Fold Cross Validation 기반의 앙상블 학습 파이프라인(`src/train_kfold.py`)을 순차적으로 실행하여 최종 가중치 5개를 생성합니다.
* **`run_tune.bat` / `run_tune.sh`**
  * 모델 성능 최적화를 위한 하이퍼파라미터 튜닝(`src/tune.py`)을 수행합니다.
* **`run_back_app.bat` / `run_back_app.sh`**
  * 수동 검토용 GUI 애플리케이션(`app_back`)을 실행합니다.
* **`run_front_app.bat` / `run_front_app.sh`**
  * 실시간 자동 검사 모니터링 GUI 애플리케이션(`app_front`)을 실행합니다.
* **`reset_to_main.bat` / `reset_to_main.sh`**
  * (초기화) 내 PC의 로컬 폴더 상태를 Github의 main 브랜치와 동일하게 초기화 및 동기화하는 스크립트입니다.

## 실행 방법

* **Windows 사용자:**
  터미널(명령 프롬프트 또는 PowerShell)에서 루트 디렉토리로 이동 후 다음과 같이 실행합니다.
  ```bat
  scripts\run_front_app.bat
  ```
* **Linux / Mac 사용자:**
  터미널에서 스크립트 파일에 실행 권한을 부여한 후 실행합니다.
  ```bash
  chmod +x scripts/*.sh
  ./scripts/run_front_app.sh
  ```
