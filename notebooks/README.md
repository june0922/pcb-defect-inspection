# notebooks — Google Colab 기반 학습 · 튜닝 노트북

GPU 서버나 Google Colab에서 모델을 학습하고 하이퍼파라미터를 튜닝하기 위한  
Jupyter Notebook 래퍼 모음입니다. Google Drive와 연동하여 데이터와 가중치를 관리합니다.

---

## 파일 구성

| 파일 | 목적 | Drive 연동 | resume 지원 |
|------|------|-----------|------------|
| `pcb_train_colab.ipynb` | 단일 모델(yolo26s) 학습 | O | O |
| `pcb_kfold_colab.ipynb` | 5-Fold K-Fold 앙상블 학습 → best_fold_1~5.pt 생성 | O | O |
| `pcb_tune_colab.ipynb` | Ray Tune 기반 하이퍼파라미터 탐색 | O | X |
| `pcb_train_tune_colab.ipynb` | 튜닝 결과를 적용한 최종 학습 | O | O |
| `pcb_kfold_tune_colab.ipynb` | 튜닝 결과를 적용한 5-Fold K-Fold 앙상블 학습 → best_fold_1~5_tune.pt 생성 | O | O |

---

## 실행 순서

```
1. pcb_tune_colab.ipynb
   └─► Ray Tune으로 최적 하이퍼파라미터 탐색
         └─► 결과를 config.yaml에 반영

2. pcb_kfold_colab.ipynb  (또는 pcb_train_colab.ipynb)
   └─► 5-Fold 학습 → weights/best_fold_1~5.pt 생성
         └─► 이 가중치가 app_front에서 사용됨

3. (선택) pcb_train_tune_colab.ipynb
   └─► 튜닝된 파라미터로 단일 모델 정밀 학습

4. (선택) pcb_kfold_tune_colab.ipynb
   └─► 튜닝된 파라미터로 5-Fold 앙상블 학습 → weights/best_fold_1~5_tune.pt 생성
```

---

## 사용 방법

1. `.ipynb` 파일을 Google Colab에 업로드하거나 GitHub에서 직접 열기
2. 런타임 유형을 **T4 GPU 이상**으로 설정
3. 첫 번째 셀부터 순서대로 실행 (Drive 마운트 → 패키지 설치 → 학습)
4. Google Drive 경로 설정 셀에서 본인 Drive 경로로 수정 후 실행

---

## 로컬 GPU 서버에서 실행할 경우

노트북 대신 `scripts/` 폴더의 스크립트를 사용하세요:

```bash
bash scripts/run_preprocess.sh   # 전처리
bash scripts/run_kfold.sh        # 5-Fold 학습
bash scripts/run_tune.sh         # 하이퍼파라미터 튜닝
```
