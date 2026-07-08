# src — 머신러닝 파이프라인 핵심 모듈

데이터 전처리부터 모델 학습, 하이퍼파라미터 튜닝까지  
프로젝트의 모든 ML 로직이 구현된 Python 모듈 모음입니다.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `preprocess.py` | DeepPCB raw 데이터 → YOLO 포맷 변환 + 그룹 단위 train/val/test 분할 |
| `merge_images.py` | 1500장 서브 이미지를 1×1~4×4 조합으로 합성 (데이터 확장) |
| `train.py` | 단일 yolo26s 모델 학습 |
| `train_kfold.py` | 5-Fold K-Fold 앙상블 학습 → `weights/best_fold_1~5.pt` 생성 |
| `tune.py` | Ray Tune 기반 하이퍼파라미터 탐색 |
| `train_tune.py` | 튜닝 결과 적용 단일 모델 정밀 학습 |
| `utils.py` | config.yaml 로드 + 환경(local/colab/server)별 경로 분기 |
| `__init__.py` | 패키지 초기화 |

---

## 전처리 파이프라인 (`preprocess.py`)

```
DeepPCB 원본
(trainval.txt + test.txt)
        │
        ▼
collect_pairs()
  └─ (image_path, label_path, group_id) 수집
     group_id = 파일명 앞 5자리 (예: "00041")
        │
        ▼
split_dataset()  — 그룹 단위 greedy 분할
  ├─ 그룹별 샘플 수 집계
  ├─ test 그룹 먼저 배정 (test.txt 기준)
  ├─ val  그룹 greedy 배정 (목표 비율까지)
  └─ 나머지 → train
     │
     ▼ 검증
  assert not (train_groups & val_groups)
  assert not (train_groups & test_groups)
        │
        ▼
save_yolo_format()
  ├─ 이미지 → preprocessed_data/{train,val,test}/images/
  └─ 라벨  → preprocessed_data/{train,val,test}/labels/
             (DeepPCB "x1 y1 x2 y2 type" → YOLO "cls cx cy w h" 정규화)
```

---

## KFold 그룹 격리 검증 결과

**결론: 그룹 데이터 누수 없음 — 코드 수정 불필요.**

두 겹의 보호 레이어로 test 그룹이 학습에 절대 유입되지 않습니다:

### 레이어 1 — 파일시스템 격리

`preprocess.py`의 `split_dataset()`이 그룹을 `train/val/test` 폴더에 완전 격리하며,  
`assert not (tg & vg) and not (tg & teg)` 로 교차 검증합니다.  
`train_kfold.py`는 `for split in ["train", "val"]:` 로 `test/` 폴더를 명시적으로 제외합니다.

### 레이어 2 — KFold 분할 격리

`StratifiedGroupKFold(groups=[stem[:5]])` 를 사용하여  
동일 그룹 ID가 같은 fold의 train/val 양쪽에 동시 등장하지 않도록 보장합니다.

---

## 학습 파이프라인

```
[단일 학습]        scripts/run_train.bat → src/train.py
                       └─► weights/best.pt

[5-Fold 앙상블]    scripts/run_kfold.bat → src/train_kfold.py
                       ├─► weights/best_fold_1.pt
                       ├─► weights/best_fold_2.pt
                       ├─► weights/best_fold_3.pt
                       ├─► weights/best_fold_4.pt
                       └─► weights/best_fold_5.pt  ← app_front/app_back 사용

[하이퍼파라미터]   scripts/run_tune.bat  → src/tune.py (Ray Tune)
                       └─► 최적 파라미터 → config.yaml 반영

[튜닝 후 학습]     scripts/run_train_tune.bat → src/train_tune.py
```

---

## utils.py — 환경 분기

`config.yaml`의 `env` 키 값에 따라 경로를 자동 분기합니다.

| env 값 | 대상 환경 | raw_data 경로 |
|--------|-----------|--------------|
| `local` | 개발 PC | `config.yaml` 내 `paths.local.raw_data` |
| `colab` | Google Colab | `/content/drive/MyDrive/...` |
| `server` | GPU 서버 | `/shared/...` |

```python
cfg = load_config("config.yaml")
paths = get_paths(cfg)
# paths["raw_data"], paths["processed"], paths["weights"], paths["runs"]
```
