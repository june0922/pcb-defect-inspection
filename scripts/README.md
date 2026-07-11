# scripts — 전처리 · 학습 · 초기화 실행 스크립트

전처리, 학습, 하이퍼파라미터 튜닝, 환경 초기화를 간편하게 실행하는  
배치 파일(`.bat`)과 쉘 스크립트(`.sh`) 모음입니다.

> **앱 실행 스크립트**는 이 폴더가 아닌 각 앱 폴더(`app_front/`, `app_back/`)에 있습니다.  
> 각각의 `run_app.bat` (Windows) 또는 `run_app.sh` (macOS/Linux)를 독립적으로 실행하세요.

---

## 스크립트 목록

모든 스크립트는 **프로젝트 루트**에서 실행합니다.

| 스크립트 | 동작 |
|----------|------|
| `run_preprocess.bat` / `.sh` | `src/preprocess.py` 실행 — 원본 DeepPCB 데이터를 YOLO 포맷으로 변환 + train/val/test 분할 |
| `run_train.bat` / `.sh` | `src/train.py` 실행 — 단일 yolo26s 모델 학습 |
| `run_kfold.bat` / `.sh` | `src/train_kfold.py` 실행 — 5-Fold K-Fold 앙상블 학습 → `weights/best_fold_1~5.pt` 생성 |
| `run_tune.bat` / `.sh` | `src/tune.py` 실행 — 하이퍼파라미터 탐색 |
| `run_train_tune.bat` / `.sh` | `src/train_tune.py` 실행 — 튜닝 결과 적용 후 단일 모델 정밀 학습 |
| `run_kfold_tune.bat` / `.sh` | `src/train_kfold_tune.py` 실행 — 튜닝 결과 적용 5-Fold 앙상블 학습 → `weights/best_fold_1~5_tune.pt` 생성 |
| `reset_to_main.bat` / `.sh` | **주의: 전체 초기화** — 로컬 저장소를 `origin/main`과 동일하게 강제 리셋 |

---

## 유틸리티 스크립트

| 파일 | 역할 |
|------|------|
| `show_config.py` | `config.yaml` 내용을 출력 (환경 설정 확인용) |
| `get_base_model.py` | `weights/yolo26s.pt` (학습 베이스 모델) 다운로드 |

---

## 실행 방법

**Windows:**
```bat
scripts\run_preprocess.bat
scripts\run_kfold.bat
```

**macOS / Linux:**
```bash
bash scripts/run_preprocess.sh
bash scripts/run_kfold.sh
```

---

## 공통 동작

모든 스크립트는 다음 순서로 동작합니다:

1. 프로젝트 루트로 이동 (`%~dp0..` / `$(dirname "$0")/..`)
2. `venv/` 가상환경 감지 후 자동 활성화 (없으면 현재 Python 환경 사용)
3. `show_config.py`로 현재 `config.yaml` 설정 값 출력
4. (`run_preprocess` 제외) 전처리 데이터(`preprocessed_data/images/train`) 존재 여부 확인 — 없으면 에러 후 종료
5. 진행 여부를 묻는 Y/N 프롬프트 (기본값 N — 그냥 Enter 시 취소)
6. 해당 Python 모듈 실행

**이어학습(Resume) 프롬프트** — 학습/튜닝 스크립트는 이전 결과물이 남아있으면 실행 전 추가로 묻습니다.
- `run_train` / `run_train_tune`: `runs/train/weights/last.pt`가 있으면 Y/N으로 이어학습 여부 확인 (`--resume` 플래그 전달)
- `run_tune`: `runs/tune/`이 있으면 R(이전 iteration부터 이어서)/O(`runs/tune/` 삭제 후 재시작)/N(취소, 기본값) 중 선택
- `run_kfold` / `run_kfold_tune`: `runs/kfold(_tune)/`이 있으면 R(완료된 fold는 건너뛰고 중단된 fold는 `last.pt`에서 이어학습, `--resume` 전달)/O(모든 fold 처음부터)/N(취소, 기본값) 중 선택

---

## reset_to_main 경고

```
⚠️  reset_to_main.bat / .sh
    이 스크립트는 uncommitted 변경 사항 및 untracked 파일을 모두 삭제합니다.
    실행 전 반드시 작업 내용을 커밋하거나 백업하세요.
```

`git reset --hard`/`git clean -fdx` 실행 전, 아래 디렉터리를 먼저 강제 삭제합니다(Windows 파일 잠금으로 인한 프롬프트 멈춤 방지 목적).
`preprocessed_data/`, `dataset/`, `venv/`, `src/__pycache__/`, `app_back/__pycache__/`, `web_test/preprocessed_data/`, `web_test/results/`, `web_test/runs/`, `web_test/weights/`
