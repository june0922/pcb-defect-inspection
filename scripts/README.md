# scripts — 전처리 · 학습 · 초기화 실행 스크립트

전처리, 학습, 하이퍼파라미터 튜닝, 환경 초기화를 간편하게 실행하는  
배치 파일(`.bat`)과 쉘 스크립트(`.sh`) 모음입니다.

> **앱 실행 스크립트**는 이 폴더가 아닌 `app_front/` 에 있습니다.  
> `app_front/run_app.bat` (Windows) 또는 `app_front/run_app.sh` (macOS/Linux)를 실행하면  
> **app_back + app_front가 순서대로 자동 시작**됩니다.

---

## 스크립트 목록

모든 스크립트는 **프로젝트 루트**에서 실행합니다.

| 스크립트 | 동작 |
|----------|------|
| `run_preprocess.bat` / `.sh` | `src/preprocess.py` 실행 — 원본 DeepPCB 데이터를 YOLO 포맷으로 변환 + train/val/test 분할 |
| `run_train.bat` / `.sh` | `src/train.py` 실행 — 단일 yolo26s 모델 학습 |
| `run_kfold.bat` / `.sh` | `src/train_kfold.py` 실행 — 5-Fold K-Fold 앙상블 학습 → `weights/best_fold_1~5.pt` 생성 |
| `run_tune.bat` / `.sh` | `src/tune.py` 실행 — Ray Tune 기반 하이퍼파라미터 탐색 |
| `run_train_tune.bat` / `.sh` | `src/train_tune.py` 실행 — 튜닝 결과 적용 후 단일 모델 정밀 학습 |
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
3. 해당 Python 모듈 실행

---

## reset_to_main 경고

```
⚠️  reset_to_main.bat / .sh
    이 스크립트는 uncommitted 변경 사항 및 untracked 파일을 모두 삭제합니다.
    실행 전 반드시 작업 내용을 커밋하거나 백업하세요.
```
