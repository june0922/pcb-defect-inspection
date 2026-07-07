# GPU 서버 사용법 및 배포 가이드 (윈도우 / Mac 개별 가이드)

본 문서는 **3팀(team03)**의 `pcb-defect-inspection` 프로젝트를 GPU 서버(Ubuntu 기반)에서 배포하고 실행하는 구체적인 절차를 정리한 가이드라인입니다. 

---

## 💡 Mac과 Linux의 터미널 사용법은 같나요?
**네, 거의 같습니다.** 
* **Mac OS**는 UNIX 기반 시스템(Darwin Kernel)을 사용하고 있고, **Linux(Ubuntu 등)** 또한 UNIX 계열 시스템입니다. 
* 따라서 터미널을 다루는 기본 명령어(예: `ssh`, `cd`, `ls`, `scp`, `git`, `mkdir` 등)와 쉘 환경(Mac의 `zsh`, Ubuntu의 `bash`)의 문법이 사실상 완전히 호환됩니다.
* 단, 윈도우는 DOS/NT 커널을 사용하므로, 터미널 환경(CMD/PowerShell)과 파일 전송 방식 등에서 Mac/Linux와 다른 점들이 존재합니다. 본 문서에서는 이러한 차이점을 고려하여 **윈도우**와 **Mac(Linux 공통)** 사용자를 위한 가이드를 분리하여 작성했습니다.

---

## 🌐 공통 서버 정보 (접속/포트/경로)

서버 접속 및 사용에 필요한 공통 메타데이터입니다.

* **서버 IP 주소:** `165.246.170.53`
* **접속 프로토콜:** SSH (기본 포트 22)
* **3팀 계정명 (ID):** `team03`
* **3팀 홈 디렉토리:** `/data/inha/team03`
* **3팀 작업 폴더 (Workspace):** `/data/inha/team03/workspace` (모든 코드는 이 아래에 보관해야 합니다)
* **3팀 주피터 권장 포트:** `8883`
* **공용 데이터 및 모델 경로:** `/data/inha/shared`

---

## 🪟 1. 윈도우 (Windows) 사용자 가이드

윈도우 환경에서는 기본 명령 프롬프트(CMD), PowerShell 또는 Git Bash를 사용해 서버에 접속하고 제어할 수 있습니다.

### [Step 1] GPU 서버 SSH 접속
1. **PowerShell** 또는 **CMD** 창을 실행합니다. (혹은 설치된 **Git Bash**를 사용해도 좋습니다.)
2. 아래 명령어를 입력하고 엔터를 누릅니다.
   ```bash
   ssh team03@165.246.170.53
   ```
3. 비밀번호 입력 프롬프트가 나타나면 제공된 **초기 비밀번호**를 입력합니다. (보안상 입력하는 비밀번호 글자는 화면에 표시되지 않으니 그대로 타이핑 후 엔터를 치시면 됩니다.)
4. *최초 로그인 시 비밀번호 변경을 요구받을 수 있습니다. 기존 비밀번호 입력 후 새로운 비밀번호로 변경 절차를 완료합니다.*

### [Step 2] 작업 디렉토리 이동 및 프로젝트 Clone
접속에 성공하면 아래 순서대로 명령어를 실행해 Git 저장소를 클론합니다.
```bash
# 3팀 작업 공간으로 이동
cd ~/workspace

# 저장소 클론 (깃허브 주소)
git clone https://github.com/june0922/pcb-defect-inspection.git

# 프로젝트 폴더로 진입
cd pcb-defect-inspection
```

### [Step 3] 대용량 데이터셋(LFS) 동기화
윈도우 환경에서 local 작업을 푸시했거나 LFS 파일이 제대로 다운로드되지 않고 포인터 파일(수십 KB 수준)로만 존재할 경우, 서버 터미널 내에서 아래 명령을 실행하여 원본 zip 파일을 완전히 내려받습니다.
```bash
git lfs pull
```

### [Step 4] Conda 가상환경 활성화 및 의존성 패키지 설치
서버에 세팅되어 있는 공용 가상환경 `ai`를 사용합니다.
```bash
# Conda 환경 설정 적용
source ~/.bashrc

# ai 가상환경 활성화
conda activate ai

# 터미널 프롬프트 앞에 (ai) team03@...이 표시되는지 확인합니다.

# 프로젝트에 필요한 패키지 추가 설치
pip install -r requirements.txt
```

### [Step 5] 설정 파일 확인 및 데이터 전처리
1. 프로젝트의 기본 설정 파일인 `config.yaml`을 확인하거나 필요시 수정합니다.
   * `env: local` 옵션 하위에 있는 `project_root: .`, `raw_data: dataset/PCBData` 등은 상대 경로로 세팅되어 있어 수정 없이 바로 작동합니다.
   * 설정을 변경하고 싶다면 리눅스 편집기(`nano` 등)를 이용해 편집합니다.
     ```bash
     nano config.yaml
     ```
2. 대용량 파일 `dataset.zip`을 해제하고 YOLO 포맷으로 변환하는 전처리 파이프라인을 실행합니다.
   ```bash
   python src/preprocess.py
   ```
   * 완료 후 `preprocessed_data/` 디렉토리에 학습용 파티셔닝 데이터가 구축됩니다.

### [Step 6] GPU 상태 확인 및 학습 진행
1. 현재 다른 팀이 사용 중인 GPU 상태를 점검합니다.
   ```bash
   nvidia-smi
   ```
2. 3팀에 배정된 GPU 일정을 확인한 뒤, 지정된 GPU 카드를 사용해 학습을 구동합니다.
   ```bash
   # 0번 GPU 카드를 지정하여 학습 실행
   CUDA_VISIBLE_DEVICES=0 python src/train.py
   ```
   * 학습 완료 시 최종 가중치는 `weights/best.pt` 경로에 자동으로 저장됩니다.

### [Step 7] Jupyter Lab 원격 연결 (포트 포워딩)
1. **서버 터미널**에서 아래 명령을 입력하여 Jupyter Lab을 실행합니다. (권장 포트: 8883)
   ```bash
   jupyter lab --port 8883 --no-browser
   ```
2. **로컬 PC(내 윈도우 컴퓨터)**에서 **새로운 PowerShell/CMD 창**을 하나 더 엽니다.
3. 아래 포트 포워딩 명령어를 실행해 터널을 뚫어줍니다.
   ```cmd
   ssh -L 8883:localhost:8883 team03@165.246.170.53
   ```
4. 웹 브라우저(Chrome, Edge 등)를 열고 주소창에 `http://localhost:8883`을 입력하여 접속합니다. (최초 접속 시 서버 터미널 창에 뜬 token 문자열이 필요할 수 있습니다.)

### [Step 8] 파일 전송 (로컬 PC ↔ GPU 서버)
윈도우 환경에서 서버와 대용량 파일이나 학습 로그를 편리하게 주고받으려면 SFTP GUI 클라이언트를 쓰는 것을 추천합니다.
* **추천 프로그램:** **WinSCP** 또는 **FileZilla**
* **연결 설정:**
  * 호스트 이름: `165.246.170.53`
  * 포트 번호: `22` (SFTP 프로토콜)
  * 사용자명: `team03`
  * 비밀번호: *설정하신 비밀번호*
* 연결 후 내 PC 폴더와 서버의 `/data/inha/team03/workspace/pcb-defect-inspection/` 폴더 간 드래그 앤 드롭으로 간편하게 전송할 수 있습니다.
* PowerShell 명령어로 직접 전송 시:
  ```cmd
  # 예: 로컬 weights 폴더의 파일을 서버로 업로드
  scp weights/best.pt team03@165.246.170.53:~/workspace/pcb-defect-inspection/weights/
  ```

---

## 🍏 2. Mac 및 Linux 사용자 가이드

Mac 및 Linux 환경은 OS 자체에 강력한 UNIX 터미널이 내장되어 있어, 별도의 타사 프로그램 설치 없이 터미널 하나로 효율적인 작업이 가능합니다.

### [Step 1] GPU 서버 SSH 접속
1. Mac의 **터미널(Terminal)** 또는 **iTerm2**를 엽니다.
2. 아래 명령어를 실행하여 서버로 로그인합니다.
   ```bash
   ssh team03@165.246.170.53
   ```
3. 패스워드를 입력하여 로그인을 완료합니다.

### [Step 2] 작업 디렉토리 이동 및 프로젝트 Clone
```bash
# 3팀 작업 디렉토리 이동
cd ~/workspace

# 깃 레포지토리 복제
git clone https://github.com/june0922/pcb-defect-inspection.git

# 폴더 이동
cd pcb-defect-inspection
```

### [Step 3] Git LFS 파일 내려받기
`dataset.zip` 파일이 용량 제한으로 LFS 포인터로 복사되어 있을 수 있으므로 풀링 작업을 해줍니다.
```bash
git lfs pull
```

### [Step 4] 가상환경 활성화 및 추가 라이브러리 검증
```bash
# 환경 설정 로드
source ~/.bashrc

# 가상환경 켜기
conda activate ai

# (ai) team03@... 프롬프트가 나타나면 의존성 설치
pip install -r requirements.txt
```

### [Step 5] 설정 확인 및 전처리 파이프라인
```bash
# 설정 변경이 필요한 경우 nano 에디터 사용
nano config.yaml

# 데이터셋 압축 해제 및 YOLO 변환 파이프라인 구동
python src/preprocess.py
```

### [Step 6] GPU 가용성 확인 및 훈련 실행
```bash
# GPU 모니터링
nvidia-smi

# 0번(또는 1번) GPU 지정 후 백그라운드나 전면에서 학습 구동
CUDA_VISIBLE_DEVICES=0 python src/train.py
```

### [Step 7] Jupyter Lab 실행 및 원격 포트 포워딩
1. **서버 터미널**에서 Jupyter 실행:
   ```bash
   jupyter lab --port 8883 --no-browser
   ```
2. **Mac의 새로운 터미널 탭/창**을 엽니다 (`Cmd + T` 또는 `Cmd + N`).
3. Mac 터미널에서 다음 SSH 터널링 명령을 실행합니다.
   ```bash
   ssh -L 8883:localhost:8883 team03@165.246.170.53
   ```
4. Safari 또는 Chrome 등의 브라우저를 열고 `http://localhost:8883`으로 연결합니다.

### [Step 8] 파일 전송 (Mac/Linux CLI 활용)
Mac/Linux 사용자들은 터미널에서 내장 `scp` 명령어를 사용하는 것이 매우 편리합니다.
* **서버 → Mac으로 파일 다운로드 (Mac 터미널에서 실행):**
  ```bash
  # 서버의 학습 결과(weights/best.pt)를 내 Mac의 현재 폴더(.)로 가져오기
  scp team03@165.246.170.53:~/workspace/pcb-defect-inspection/weights/best.pt ./
  
  # 서버의 runs 디렉토리 전체를 Mac으로 가져오기 (-r 옵션 사용)
  scp -r team03@165.246.170.53:~/workspace/pcb-defect-inspection/runs ./
  ```
* **Mac → 서버로 파일 업로드 (Mac 터미널에서 실행):**
  ```bash
  # 로컬 수정 코드를 서버로 업로드
  scp src/train.py team03@165.246.170.53:~/workspace/pcb-defect-inspection/src/
  ```
* GUI 환경을 선호할 경우, **Cyberduck** 또는 **FileZilla** 프로그램을 다운로드하여 SFTP 연결(포트 22)을 사용하면 됩니다.

---

## ⚠️ GPU 실습 서버 공통 사용 규칙 (필독)

1. **상시 사용 전 모니터링:** 훈련 구동 전에 무조건 `nvidia-smi`를 실행하여 다른 프로세스의 VRAM 점유 상태를 체크하고 간섭이 발생하지 않도록 해야 합니다.
2. **지정 GPU 준수:** 3팀에게 할당된 GPU 디바이스 번호(0번 또는 1번)를 환경변수(`CUDA_VISIBLE_DEVICES=번호`)를 통해 꼭 선언해 실행해야 타 팀의 훈련 프로세스를 방해하지 않습니다.
3. **용량 확보:** 대용량 모델 파일, 에폭별 체크포인트, 텐서보드 로그 등은 학습 및 분석이 끝나는 대로 압축 또는 불필요한 캐시 정리를 해 주어야 전체 서버 디스크 용량 초과를 막을 수 있습니다.
4. **결과물 백업:** 중요 체크포인트 가중치(`best.pt`)와 학습 데이터 분석 결과는 로컬 컴퓨터나 GitHub 원격 저장소에 수시로 백업 및 복사해 두시기 바랍니다.
