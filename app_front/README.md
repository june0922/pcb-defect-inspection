# app_front — 실시간 자동 PCB 결함 검사 모니터링

PCB 이미지를 설정 가능한 크기(기본 640×640)의 타일로 분할하여, Options에서 선택한 1개 이상(기본값 5개, K-Fold 학습)의 YOLO26 모델의 WBF 앙상블 추론으로 결함을 자동 검출합니다.  
판정 결과(PASS/FAIL/REVIEW)는 SQLite DB에 실시간 저장되며, `app_back`이 REVIEW 타일을 수신합니다.

---

## 파일 구성

```
app_front/
├── run.py                    # 진입점 (다크 테마 적용 후 MainWindow 실행)
├── run_app.bat               # Windows: app_front 단독 실행 스크립트
├── run_app.sh                # macOS/Linux: app_front 단독 실행 스크립트
├── main_ui.py                # 메인 윈도우 (MainWindow, SettingsDialog, DefaultsEditDialog, DatabaseDialog)
├── inspection_worker.py      # 타일 검사 워커 스레드 (InspectionWorker)
├── global_view.py            # PCB 전체 뷰어 + 현재 검사 위치 스캔박스 (GlobalView)
├── vision_viewer.py          # 타일 확대 뷰어 + 결함 오버레이 (VisionViewer, 읽기전용)
├── class_band_table.py       # 클래스별 REVIEW MIN/MAX 추가/삭제 테이블 위젯 (ClassBandTableWidget)
├── model_paths_widget.py     # 검사 모델(.pt) 선택/검증 위젯 (ModelPathsWidget)
├── defaults_store.py         # 공장 기본값(default_settings.json) 읽기/쓰기 + DB 강제 리셋
├── default_settings.json     # 공장 기본값 (클래스, 타일크기, 오버랩, 모델 경로 등)
└── models/                   # 기본 검사 모델 (best_fold_1_tune~5_tune.pt) 배포 위치
```

---

## 실행 방법

```bash
# Windows
app_front\run_app.bat
# macOS/Linux
bash app_front/run_app.sh

# 또는 직접 실행
cd app_front
python run.py
```

app_front와 app_back(`app_back/run_app.bat` 또는 `.sh`)은 각자 독립적으로 실행되는
별개의 프로그램입니다 — 어느 쪽을 먼저 켜도, 하나만 켜도 정상 동작합니다.

---

## 메뉴 구조

메뉴바의 세 항목은 드롭다운 없이 **클릭 1번으로 바로 동작**한다.

| 메뉴 | 단축키 | 동작 |
|------|--------|------|
| **File** | `Ctrl+O` | 바로 폴더 선택 대화상자를 열고 검사 시작 |
| **Database** | — | `DatabaseDialog`를 바로 열어 통계 조회 + 하단 "DB 초기화" 버튼 제공(같은 창에서 즉시 0으로 갱신) |
| **Option** | — | `SettingsDialog`를 바로 열어 클래스별 REVIEW 밴드 / 타일크기·오버랩 / 검사 모델 / 경고음 설정 |

---

## 단축키

| 키 | 동작 |
|----|------|
| `Ctrl+O` | 폴더 열기 |
| `Space` | 검사 일시정지 / 재개 |

---

## Settings Dialog (Option)

클래스별로 독립적인 REVIEW 판정 구간을 설정합니다.

```
┌──────────────┬──────────────────┬──────────────────┐
│ 결함 클래스   │ REVIEW MIN (%)   │ REVIEW MAX (%)   │
├──────────────┼──────────────────┼──────────────────┤
│ open         │  [SpinBox 1-99]  │  [SpinBox 2-100] │
│ short        │        …         │        …         │
│  …           │        …         │        …         │
└──────────────┴──────────────────┴──────────────────┘
[클래스 추가 [+]]  [클래스 삭제 [-]]   ← 클래스를 자유롭게 추가/삭제 (최소 1개 유지)

타일 크기: [8~1280px]     오버랩: [0~90%]
검사 모델 (.pt 1개 이상, 앙상블 추론에 사용): [선택된 모델 N개: ...]  [모델 파일 선택...]
[✓] FAIL 검출 시 경고음
[기본값 수정...]  ← 프로그램 재시작 시 적용될 공장 기본값(default_settings.json) 자체를 수정
```

- **MIN 미만**: PASS (초록 — LocalView에서 표시, `_GREEN_FLOOR`=1% 이상부터)
- **MIN 이상 MAX 이하**: REVIEW (app_back에서 수동 검토, 노랑)
- **MAX 초과**: FAIL (불량 확정, 빨강)
- **값을 변경하면 폴더를 재오픈한 것처럼 동작한다**: 실제로 값이 바뀌었고 폴더가 이미 열려 있으면 확인창 후 DB를 초기화하고 처음부터 다시 검사한다. 즉시 적용/다음 이미지부터 적용되는 방식이 아니다.

---

## 타일 처리 파이프라인

```
[폴더 선택]
     │
     ▼
[이미지 로드] ── _colored.png 제외 ── 이진화 원본만 사용
     │
     ▼
[Pre-scan]  compute_grid(h, w, tile_size, overlap_pct) → total_tiles 사전 집계
     │
     ▼
[서펜타인(ㄹ자) 스캔]  serpentine_order()
  ┌──────────────────────────────────────────┐
  │ 짝수 이미지: 위→아래, 행마다 방향 교대     │
  │ 홀수 이미지: 아래→위 역방향 시작          │
  │ (인접 PCB 간 연속성 유지)                │
  └──────────────────────────────────────────┘
     │
     ▼ (row, col) 순서로 반복
[타일 추출]  _extract_tile(img, row, col)
  ├─ tile_size×tile_size 크롭 (Options에서 설정, 기본 640×640)
  └─ 경계 초과 시 흰색(255) 패딩
     │
     ▼
[WBF 앙상블 추론]  _ensemble_predict(tile_bgr)
  ├─ 선택된 YOLO26 모델(1개 이상) 각각 predict(conf=global_floor)
  │    global_floor = min(모든 클래스의 review_min)
  ├─ weighted_boxes_fusion(iou_thr, skip_box_thr=global_floor)
  └─ 활성 클래스 목록에 없는 검출만 제거 (review_min 미만도 남겨 LocalView에서 초록으로 표시)
     │
     ▼
[판정]  _classify_verdict(detections) / _classify_detection_verdict(det)
  ├─ 타일 전체 요약(verdict): 검출 없음 → PASS, FAIL > REVIEW > PASS 우선순위
  └─ 결함 1건씩(_classify_detection_verdict): 개별 conf 기준 REVIEW/FAIL/PASS 판정
     │
     ▼
[DB 저장]  upsert_tile(tile_bgr, verdict, image_path, row, col, detections)
  ├─ tiles: UNIQUE(image_path, grid_row, grid_col) — 동일 위치 재처리 시 최신값으로 교체
  └─ defects: REVIEW/FAIL 등급 결함만 bbox/class/confidence/개별 verdict와 함께 저장
     (PASS 등급은 저장하지 않음 — app_back이 재추론 없이 그대로 읽어 표시)
     │
     ▼
[UI 업데이트]
  ├─ GlobalView: 실제 픽셀 좌표에 현재 검사 위치 스캔박스 갱신(테두리+반투명 채우기)
  │    + FAIL 판정 타일에는 영구 X 마커 추가(add_fail_marker)
  ├─ VisionViewer: 타일 이미지 + 결함 오버레이(클래스별 REVIEW 밴드 기준 초록/노랑/빨강)
  ├─ FilmStrip: 썸네일 추가 (최근 30개만 유지, 초과분은 자동 제거)
  └─ Statistics Panel: 통계 갱신
```

---

## UI 레이아웃

```
┌─────────────────────────────────────────────────────┐
│ File  Database  Option                              │  ← 메뉴바
├───────────────────────┬─────────────────────────────┤
│   GlobalView (30%)    │    VisionViewer (70%)        │
│                       │                             │
│  컬러 PCB 전체 이미지  │  현재 타일 (이진화)           │
│  + 현재 위치 스캔박스  │  + 결함 코너 브라켓 오버레이  │
│  (해상도 무관 핏)      │                             │
│  Statistics Panel     │                             │
│  ─────────────────    │                             │
│  Inspected: N/total   │                             │
│  PASS:  N (X.X%)     │                             │
│  FAIL:  N (X.X%)     │                             │
│  REVIEW: N (X.X%)    │                             │
│  FPY:   X.X%         │                             │
│  Throughput: X t/min │                             │
│  Elapsed: HH:MM:SS   │                             │
│  Defects: open:N ...  │                             │
├───────────────────────┴─────────────────────────────┤
│              FilmStrip (썸네일 가로 스크롤)            │
└─────────────────────────────────────────────────────┘
│ StatusBar: 상태 메시지           [Progress Bar]      │
```

---

## 주요 컴포넌트

### GlobalView (`global_view.py`)

컬러 PCB 전체 이미지 위에 현재 검사 위치만 실제 픽셀 좌표 기준 박스 1개로 표시합니다(그리드/셀 상태 개념은 없음 — 오버랩 설정에 따라 타일이 서로 겹칠 수 있어 셀 단위 색상 표시는 의미가 없다).

- `set_image()`: `fitInView(Qt.KeepAspectRatio)`로 이미지의 절대 해상도와 무관하게 항상 뷰포트를 종횡비 유지하며 꽉 채운다. 이진화 원본과 짝을 이루는 `{stem}_colored.png`가 없으면, 스테일 화면을 남기지 않도록 이진화 원본 자체를 3채널로 변환해 폴백으로 표시한다.
- `set_scan_box(x, y, size)`: 타일 좌상단 실제 픽셀 좌표(`col*stride, row*stride`, `InspectionWorker._compute_stride()`와 동일한 오버랩 스트라이드 공식)에 빨간 테두리(`QColor(255,40,40,230)`) + 반투명 빨간 채우기(`QColor(255,40,40,64)`)로 표시. 다음 타일 검사 시 이전 위치 스캔박스는 사라진다(일시적).
- `add_fail_marker(x, y, size)`: 완전 FAIL 판정 타일 위치 중앙에 불투명 빨간 X 표식(`QColor(255,20,20,255)`)을 영구적으로 남긴다. 스캔박스와 달리 `set_image()`/`clear_all()` 전까지 계속 유지되어, 검사가 끝난 뒤에도 PCB 전체에서 FAIL 타일 위치를 한눈에 확인할 수 있다.
- 셀 단위 PASS/FAIL/REVIEW 색상 표시(그리드 채우기)는 GlobalView에서 제거되었다 — 실시간 통계는 별도의 텍스트 기반 Statistics Panel에서 확인하고, GlobalView는 현재 검사 위치(스캔박스)와 FAIL 확정 위치(X 마커)만 시각적으로 표시한다.

### VisionViewer (`vision_viewer.py`, 읽기전용)

현재 처리 중인 타일을 표시하는 비대화형 뷰어입니다.
- 결함 위치에 **코너 브라켓** 오버레이 (팔 길이 = bbox 너비의 20%)
- 라벨: `{class_name} {conf:.2f}` — 줌 불변 크기 (Consolas 9pt Bold, 검정 70% 불투명 배경)
- 신뢰도 색상: 빨강(FAIL 수준) / 노랑(REVIEW 수준) / 초록(PASS 수준) — 클래스별 REVIEW MIN/MAX(%)에 동적으로 연동

---

## 모델 앙상블 구조

Options(`ModelPathsWidget`)에서 `.pt` 파일을 **1개 이상** 자유롭게 선택할 수 있다(개수 상한 없음). 기본값은 `app_front/models/`에 배포된 5-Fold 학습 모델이다.

```
app_front/models/            (기본값 — Options에서 다른 .pt로 자유롭게 교체 가능)
├── best_fold_1_tune.pt  ┐
├── best_fold_2_tune.pt  │
├── best_fold_3_tune.pt  ├─► 5-Fold 학습 결과, WBF 앙상블로 함께 추론 (Yes-Tune)
├── best_fold_4_tune.pt  │
└── best_fold_5_tune.pt  ┘
```

- 선택된 모델들이 각각 독립적으로 추론 → `weighted_boxes_fusion()` 으로 박스 병합(모델 개수와 무관하게 동작)
- 단순 NMS보다 재현율이 높고 다수결 기반으로 안정적인 검출
- 모델 선택 시 4단계 검증(확장자 `.pt`, 로드+더미 추론 성공, 2개 이상이면 `class names` 동일성, 활성 클래스와 최소 1개 이상 이름 겹침)을 통과해야 하며, 하나라도 실패하면 경고 후 이전 선택을 유지한다.

---

## 의존성

```
PyQt5 ≥ 5.15
ultralytics    # YOLO26
ensemble_boxes # WBF (weighted_boxes_fusion)
opencv-python
numpy
torch          # GPU 가속 시
```
