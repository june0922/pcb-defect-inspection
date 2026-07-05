# Weights Directory

본 `weights` 폴더는 프로젝트에서 사용되는 **YOLOv8 모델의 가중치(Weight) 파일**들을 저장하고 관리하는 공간입니다. 

## 포함된 주요 파일

* **`yolov8n.pt`**
  * 모델 학습의 초기화에 사용되는 Ultralytics의 사전 학습된(Pre-trained) 기반 가중치(Base Weight) 파일입니다.
* **`best_fold_1.pt` ~ `best_fold_5.pt`**
  * K-Fold 교차 검증 파이프라인(`src/train_kfold.py`)을 통해 학습된 5개의 개별 모델 가중치입니다.
  * 각각 다른 데이터 부분집합에 대해 검증(Validation)을 수행하여 최고 성능(Best)을 낸 가중치가 저장됩니다.

## 주요 역할

1. **앙상블 추론(Ensemble Inference):**
   * GUI 애플리케이션(`app` 폴더 내 리뷰 스테이션) 구동 시, 내부적으로 이 폴더에 있는 5개의 가중치(`best_fold_1.pt` ~ `5.pt`)를 한꺼번에 불러옵니다.
   * 각 모델이 예측한 결과 박스들을 취합하고 NMS(Non-Maximum Suppression)를 적용하여 **최종 결함 탐지 결과의 신뢰성**을 극대화하는 데 사용됩니다.
2. **모델 재학습 및 튜닝:**
   * 추가 데이터셋 확보 시, 기존의 가중치를 불러와 전이 학습(Transfer Learning)이나 지속 학습(Continuous Learning)의 기반으로 활용될 수 있습니다.
