# app_back — REVIEW 타일 실시간 수신 · 수동 판정 리뷰 스테이션

`app_front`가 SQLite DB에 기록한 REVIEW 타일을 3초마다 폴링하여 수신하고,  
작업자가 키보드 단축키로 각 타일의 최종 판정(Pass/Fail 클래스)을 내립니다.  
5개 YOLO 모델(WBF 앙상블)로 재추론하여 결함 위치를 정밀 표시합니다.

---

## 파일 구성

```
app_back/
├── run.py               # 진입점 (다크 테마 적용 후 MainWindow 실행)
├── main_ui.py           # 메인 윈도우 (MainWindow, TileEntry)
├── inference_worker.py  # YOLO 모델 로딩 + 추론 워커 (InferenceWorker)
└── vision_viewer.py     # 결함 오버레이 대화형 뷰어 (VisionViewer)
```

---

## 실행 방법

```bash
# 단독 실행 (app_front와 DB를 공유하는 상태에서)
python app_back/run.py

# app_front와 동시 실행 (권장)
# Windows
app_front\run_app.bat
# macOS/Linux
bash app_front/run_app.sh
```

> app_back은 앱 시작 즉시 5개 YOLO 모델을 로딩합니다 (약 30초~1분).  
> 로딩 완료 후 DB 폴링이 자동으로 시작됩니다.

---

## 단축키

| 키 | 동작 |
|----|------|
| `Space` | 현재 타일 → **Pass** (정상/오탐 판정) |
| `1` ~ `6` | 현재 타일 → **Fail** (결함 클래스 번호: 1=open, 2=short, 3=mousebite, 4=spur, 5=copper, 6=pinhole) |
| `←` | 이전 타일로 이동 |
| `→` | 다음 타일로 이동 |
| `W` / `S` | VisionViewer 위/아래 패닝 |
| `A` / `D` | VisionViewer 좌/우 패닝 |
| `Q` / `E` | VisionViewer 축소 / 확대 |
| `Shift` (Hold) | 현재 선택된 결함 오버레이 외 나머지 숨김 |

> 판정 단축키(Space, 1~6)는 판정 즉시 다음 미검토(pending) 타일로 자동 이동합니다.

---

## DB 폴링 메커니즘

```
앱 시작
  └─► InferenceWorker 시작 (5개 YOLO 모델 비동기 로딩)
        └─► models_loaded 시그널 → QTimer(3000ms) 시작

QTimer 3초마다 _poll_db() 호출
  │
  ├─► get_settings() → DB에서 현재 설정 읽기
  │     ├─ db_session_id 변경 감지?
  │     │     └─► _on_db_reset(): filmstrip 전체 초기화,
  │     │                         _last_shown_id = 0
  │     │
  │     └─ 설정 변경 감지?
  │           └─► per_class_bands 갱신
  │               worker.min_conf = min(review_mins)
  │               worker.iou_thresh = new_iou
  │
  └─► fetch_review_tiles(after_id=_last_shown_id) → 새 타일 조회
        └─► 각 타일에 대해 _process_tile(tile_row):
              ├─ PNG BLOB → cv2.imdecode()
              ├─ worker.run_single_image_sync(img_bgr)
              │     └─ 5 YOLO 모델 추론 → WBF → detections, crops
              ├─ TileEntry 생성 (tile_id, img_bgr, detections, crops)
              └─ FilmStrip 썸네일 추가
```

---

## 판정 흐름

```
작업자 키 입력 (Space / 1~6)
  └─► _verdict_current(verdict)
        ├─ 디바운스 (100ms 이내 중복 입력 무시)
        ├─ entry.verdict = verdict
        ├─ save_user_verdict(tile_id, verdict) → DB 저장
        │     (tiles.user_verdict 컬럼 업데이트)
        ├─ 썸네일 테두리 색 갱신
        │     pending → 회색
        │     pass    → 초록
        │     1~6     → 빨강 + 번호 배지
        └─ 다음 pending 타일로 자동 이동
```

---

## UI 레이아웃

```
┌─────────────────────────────────────────────────────┐
├───────────────────────┬─────────────────────────────┤
│   GlobalView (30%)    │    VisionViewer (70%)        │
│                       │                             │
│  현재 타일 전체 미리보기│  결함 확대 뷰 (대화형)        │
│  (읽기전용 피트뷰)     │  + 코너 브라켓 오버레이       │
│                       │                             │
├───────────────────────┴─────────────────────────────┤
│         FilmStrip — REVIEW 타일 썸네일 (96×96)        │
│         가로 스크롤, 판정 상태별 테두리 색             │
├─────────────────────────────────────────────────────┤
│ StatusBar: [N/total] Pass: N | Fail: N | 미검토: N   │
└─────────────────────────────────────────────────────┘
```

---

## 주요 컴포넌트

### TileEntry

FilmStrip 1건에 대응하는 데이터 클래스.

| 속성 | 타입 | 설명 |
|------|------|------|
| `tile_id` | int | DB tiles.id (user_verdict 저장 키) |
| `img_bgr` | ndarray | 640×640 타일 BGR 이미지 |
| `detections` | list[dict] | `{class_id, class_name, confidence, bbox_abs}` 목록 |
| `crops` | list[ndarray] | 결함별 확대 크롭 이미지 |
| `verdict` | str | `"pending"` / `"pass"` / `"1"~"6"` |

### VisionViewer (`vision_viewer.py`, 대화형)

결함 오버레이를 표시하는 대화형 뷰어입니다.

| 조작 | 동작 |
|------|------|
| 우클릭 드래그 | 패닝 (이미지 이동) |
| 마우스 휠 | 줌 인/아웃 (포인터 중심) |
| 좌클릭 드래그 | 흰색 브러쉬 마스킹 |
| `Shift` | 선택 오버레이 외 숨김 |

결함 박스 색상은 `confidence_color(conf, bands)` 함수로 per-class 판단:
- **빨강**: conf > class review_max (FAIL 수준)
- **노랑**: conf ≥ class review_min (REVIEW 수준)
- **회색**: conf < class review_min (PASS 수준)

---

## 의존성

```
PyQt5 ≥ 5.15
ultralytics    # YOLOv8
ensemble_boxes # WBF (weighted_boxes_fusion)
opencv-python
numpy
torch          # GPU 가속 시
```
