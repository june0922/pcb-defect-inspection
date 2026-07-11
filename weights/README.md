# weights — 모델 가중치 파일

학습에 사용되는 베이스 모델과 5-Fold K-Fold 앙상블 최종 가중치를 저장합니다.

---

## 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `yolo26s.pt` | ~19.5 MB | YOLO26 **Small** 베이스 모델 (실제 학습/추론에 사용) |
| `best_fold_1.pt` | ~6.0 MB | 5-Fold K-Fold Fold 1 최적 가중치 |
| `best_fold_2.pt` | ~6.0 MB | 5-Fold K-Fold Fold 2 최적 가중치 |
| `best_fold_3.pt` | ~6.0 MB | 5-Fold K-Fold Fold 3 최적 가중치 |
| `best_fold_4.pt` | ~6.0 MB | 5-Fold K-Fold Fold 4 최적 가중치 |
| `best_fold_5.pt` | ~6.0 MB | 5-Fold K-Fold Fold 5 최적 가중치 |
| `best_fold_1~5_tune.pt` | ~6.0 MB × 5 | 튜닝된 하이퍼파라미터로 학습한 5-Fold K-Fold 최적 가중치 (`train_kfold_tune.py` 결과, 기존 `best_fold_1~5.pt`와 별도). **`scripts/run_kfold_tune`를 실행해야 생성되며, 기본 상태에서는 이 폴더에 존재하지 않음.** |

> `yolo26n.pt`(Nano)도 이 폴더에 남아있을 수 있으나 더 이상 사용하지 않는 레거시 파일입니다. 프로젝트는 `yolo26s.pt`(Small)만 사용합니다.

---

## 베이스 모델 설명

| 모델 | 파라미터 수 | 특징 |
|------|------------|------|
| `yolo26s.pt` (Small) | ~11.2M | 속도-정확도 균형, **실제 프로젝트 사용** |

`config.yaml`에는 전역 `model_size` 키가 없습니다 — `train`/`train_tune`/`tune`/`kfold`/`kfold_tune` 각 섹션이 개별적으로 `model: weights/yolo26s.pt` 키를 갖고 베이스 모델 경로를 지정합니다.  
현재 모든 섹션이 동일하게 **yolo26s** (Small)를 가리킵니다.

베이스 모델이 없는 경우 `scripts/get_base_model.py`로 다운로드합니다:
```bash
python scripts/get_base_model.py
```

---

## best_fold_1~5.pt

5-Fold K-Fold 학습 완료 후 각 Fold의 `best.pt`를 이 폴더에 복사한 파일입니다.

- **app_front**: `InspectionWorker`가 시작 시 5개 모델을 모두 로드하여 타일 추론에 사용
- WBF(Weighted Box Fusion)로 여러 예측을 병합하여 최종 결함 위치 결정
- (참고: `app_back`은 더 이상 자체 모델을 로딩하거나 재추론하지 않음)

> **주의** — app_front는 이 폴더(`weights/`)를 직접 읽지 않습니다. 기본 설정(`app_front/default_settings.json`의 `model_paths`)은 `app_front/models/best_fold_1~5.pt`를 가리키므로, `run_kfold`로 새로 학습한 뒤 app_front에 반영하려면 `weights/best_fold_1~5.pt`를 `app_front/models/`로 **수동 복사**해야 합니다(자동 복사 스크립트 없음). 다른 경로의 가중치를 쓰려면 app_front의 `Option > Settings...`에서 직접 경로를 지정할 수 있습니다.

생성 방법:
```bash
bash scripts/run_kfold.sh   # 또는 scripts\run_kfold.bat
```

---

## best_fold_1~5_tune.pt

GA 튜닝 하이퍼파라미터(`config.yaml`의 `kfold_tune` 섹션)를 적용해 `train_kfold_tune.py`로
학습한 5-Fold 가중치입니다. 기존 `train_kfold.py`/`best_fold_1~5.pt`에는 영향을 주지 않는
별도 파이프라인의 산출물이며, 앱에서 사용하려면 `app_front` Options에서 수동으로 선택해야 합니다.

생성 방법:
```bash
bash scripts/run_kfold_tune.sh   # 또는 scripts\run_kfold_tune.bat
```
