# weights — 모델 가중치 파일

학습에 사용되는 베이스 모델과 5-Fold K-Fold 앙상블 최종 가중치를 저장합니다.

---

## 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `yolo26n.pt` | ~5.4 MB | YOLOv8 **Nano** 베이스 모델 (학습 시작점, 경량) |
| `yolo26s.pt` | ~19.5 MB | YOLOv8 **Small** 베이스 모델 (실제 학습/추론에 사용) |
| `best_fold_1.pt` | ~6.0 MB | 5-Fold K-Fold Fold 1 최적 가중치 |
| `best_fold_2.pt` | ~6.0 MB | 5-Fold K-Fold Fold 2 최적 가중치 |
| `best_fold_3.pt` | ~6.0 MB | 5-Fold K-Fold Fold 3 최적 가중치 |
| `best_fold_4.pt` | ~6.0 MB | 5-Fold K-Fold Fold 4 최적 가중치 |
| `best_fold_5.pt` | ~6.0 MB | 5-Fold K-Fold Fold 5 최적 가중치 |

---

## 베이스 모델 설명

| 모델 | 파라미터 수 | 특징 |
|------|------------|------|
| `yolo26n.pt` (Nano) | ~3.2M | 초경량, 빠른 추론 속도, 정확도 낮음 |
| `yolo26s.pt` (Small) | ~11.2M | 속도-정확도 균형, **실제 프로젝트 사용** |

`config.yaml`의 `model_size` 키로 학습 시 베이스 모델을 선택합니다.  
현재 설정: **yolo26s** (Small).

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

생성 방법:
```bash
bash scripts/run_kfold.sh   # 또는 scripts\run_kfold.bat
```
