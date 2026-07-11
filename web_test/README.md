# Web Test Directory

본 `web_test` 폴더는 **두 모델(Model A / Model B)의 test 세트 추론 결과를 나란히 비교**하기 위한 정적 웹 뷰어입니다. `app_front`/`app_back`(최종 산출물 GUI 앱)과는 무관한, 모델 성능 비교·실험용 도구입니다.

## 주요 구성 요소

* **`generate_results.py`**
  * `web_test/weights/`의 지정된 모델(단일 또는 다중 모델 WBF 앙상블)로 test 세트 이미지에 대해 실제 YOLO 추론을 수행합니다.
  * `torchmetrics`(`MeanAveragePrecision`)로 Recall, mAP@0.5, mAP@0.5:0.95를 계산합니다.
  * 추론 결과 bbox를 이미지 위에 직접 그려(`draw_boxes()`) `results/<model>/`에 저장하고, 뷰어가 읽는 `results/data.js`를 생성합니다.
  * conf/iou 임계값은 `config.yaml`이 아닌 이 파일 상단의 `CONF_THRESHOLD`/`IOU_THRESHOLD` 상수로 관리됩니다(기본값 각각 0.5, 0.45).
* **`index.html`**
  * Model A / Model B를 좌우 2-패널로 비교하는 뷰어 UI + 하단 재생 컨트롤바입니다.
* **`style.css`**
  * 뷰어 레이아웃 및 스타일시트입니다.
* **`script.js`**
  * `results/data.js`를 읽어 이미지 슬라이드쇼(이전/다음, 재생/일시정지, 배속)를 제어하고 Model A/B의 Recall·mAP 지표를 렌더링합니다. bbox는 `generate_results.py`가 이미지에 이미 그려서 저장한 결과를 그대로 표시할 뿐이며, 이 파일 자체는 bbox를 그리거나 판정(PASS/FAIL) 처리를 하지 않습니다.

## 폴더 구조

```
web_test/
├── generate_results.py   # 추론 + mAP/Recall 계산 + results/data.js 생성
├── index.html / script.js / style.css   # 비교 뷰어
├── results/               # 생성된 비교 이미지 + data.js (git 추적됨)
├── preprocessed_data/     # 평가용 test 이미지·라벨 (자동 생성, 미추적)
├── weights/                # 평가용 모델 가중치 (미추적)
└── README.md
```

> `results/` 하위에는 `generate_results.py`가 실제로 생성하는 `model_a_results`/`model_b_results` 외에, 과거 실험에서 남은 폴더(예: `baseline`, `notune` 등)가 섞여 있을 수 있습니다. 현재 유효한 비교 결과인지 확인 후 참고하세요.

## 목적

로컬 GUI 어플리케이션(`app_front`, `app_back` 폴더)과는 별개로, **웹 환경**에서 두 모델의 추론 성능(지표 + 시각적 결과)을 나란히 비교해보기 위한 실험용 도구로 활용됩니다.
