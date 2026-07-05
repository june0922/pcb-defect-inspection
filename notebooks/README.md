# Notebooks Directory

본 `notebooks` 폴더는 Google Colab과 같은 클라우드 환경에서 모델을 학습하고 하이퍼파라미터를 튜닝하기 위한 Jupyter Notebook(`.ipynb`) 파일들을 포함하고 있습니다.

## 주요 파일 안내

* **`pcb_train_colab.ipynb`**
  * 단일 모델(YOLOv8) 학습을 진행하기 위한 노트북입니다.
  * Google Drive와 마운트하여 데이터를 로드하고, 학습된 가중치(`weights`) 및 훈련 과정 로그를 저장할 수 있습니다.
* **`pcb_kfold_colab.ipynb`**
  * 5-Fold Cross Validation(교차 검증)을 적용하여 여러 개의 모델을 앙상블 학습하기 위한 노트북입니다.
  * 최종적으로 `weights` 폴더에 `best_fold_1.pt` ~ `best_fold_5.pt` 가중치를 생성하는 파이프라인이 구현되어 있습니다.
* **`pcb_tune_colab.ipynb`**
  * YOLOv8 모델의 성능을 극대화하기 위해 Ray Tune 기반의 하이퍼파라미터 최적화(Hyperparameter Tuning)를 수행하는 노트북입니다.

## 사용 방법

1. 본 폴더 내의 `.ipynb` 파일을 Google Colab에 업로드하거나, GitHub 계정과 Colab을 연동하여 파일을 엽니다.
2. 각 노트북의 첫 번째 셀부터 순서대로 실행합니다. (런타임 유형을 **T4 GPU** 이상으로 설정할 것을 권장합니다.)
3. 필요에 따라 Google Drive 경로가 설정된 부분(데이터셋 경로, 모델 저장 경로 등)을 본인 환경에 맞게 수정하여 사용하십시오.
