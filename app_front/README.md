# app_front — 실시간 자동 PCB 결함 검사 모니터링

PCB 이미지를 640×640 타일로 분할하여 5개 YOLO 모델의 WBF 앙상블 추론으로 결함을 자동 검출합니다.  
판정 결과(PASS/FAIL/REVIEW)는 SQLite DB에 실시간 저장되며, `app_back`이 REVIEW 타일을 수신합니다.

---

## 파일 구성

```
app_front/
├── run.py                # 진입점 (다크 테마 적용 후 MainWindow 실행)
├── run_app.bat           # Windows: app_back + app_front 동시 실행 스크립트
├── run_app.sh            # macOS/Linux: app_back + app_front 동시 실행 스크립트
├── main_ui.py            # 메인 윈도우 (MainWindow, SettingsDialog)
├── inspection_worker.py  # 타일 검사 워커 스레드 (InspectionWorker)
├── global_view.py        # PCB 전체 뷰어 + 그리드 오버레이 (GlobalView)
└── vision_viewer.py      # 타일 확대 뷰어 + 결함 오버레이 (VisionViewer, 읽기전용)
```

---

## 실행 방법

```bash
# 단독 실행 (app_back을 따로 켠 상태에서)
cd app_front
python run.py

# app_back과 동시 실행 (권장)
# Windows
app_front\run_app.bat
# macOS/Linux
bash app_front/run_app.sh
```

---

## 메뉴 구조

| 메뉴 | 항목 | 단축키 | 동작 |
|------|------|--------|------|
| **File** | Open Folder... | `Ctrl+O` | 검사 폴더 선택 후 검사 시작 |
| **Database** | DB 통계... | — | 판정별 타일 수 + DB 파일 크기 조회 |
| **Database** | DB 초기화... | — | tiles 테이블 전체 삭제 (실행 중 Worker 유지) |
| **Option** | Settings... | — | 클래스별 REVIEW 밴드 / IoU / 경고음 설정 |

---

## 단축키

| 키 | 동작 |
|----|------|
| `Ctrl+O` | 폴더 열기 |
| `Space` | 검사 일시정지 / 재개 |

---

## Settings Dialog (Option > Settings...)

클래스별로 독립적인 REVIEW 판정 구간을 설정합니다.

```
┌──────────────┬──────────────────┬──────────────────┐
│ 결함 클래스   │ REVIEW MIN (%)   │ REVIEW MAX (%)   │
├──────────────┼──────────────────┼──────────────────┤
│ open         │  [SpinBox 1-99]  │  [SpinBox 2-100] │
│ short        │        …         │        …         │
│ mousebite    │        …         │        …         │
│ spur         │        …         │        …         │
│ copper       │        …         │        …         │
│ pinhole      │        …         │        …         │
└──────────────┴──────────────────┴──────────────────┘
IoU Threshold:  [0.10 ~ 0.95, step 0.05]  기본값: 0.45
[✓] FAIL 검출 시 경고음
```

- **MIN 미만**: PASS (정상으로 처리)
- **MIN 이상 MAX 이하**: REVIEW (app_back에서 수동 검토)
- **MAX 초과**: FAIL (불량 확정)
- 설정 변경은 즉시 DB에 저장, 현재 실행 중인 Worker는 중단하지 않음

---

## 타일 처리 파이프라인

```
[폴더 선택]
     │
     ▼
[이미지 로드] ── _colored.png 제외 ── 이진화 원본만 사용
     │
     ▼
[Pre-scan]  compute_grid(h/640, w/640) → total_tiles 사전 집계
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
  ├─ 640×640 크롭
  └─ 경계 초과 시 흰색(255) 패딩
     │
     ▼
[WBF 앙상블 추론]  _ensemble_predict(tile_bgr)
  ├─ 선택된 YOLO 모델(1개 이상) 각각 predict(conf=global_floor)
  │    global_floor = min(모든 클래스의 review_min)
  ├─ weighted_boxes_fusion(iou_thr, skip_box_thr=global_floor)
  └─ 활성 클래스 목록에 없는 검출만 제거 (review_min 미만도 남겨 LocalView에서 초록으로 표시)
     │
     ▼
[판정]  _classify_verdict(detections)
  ├─ 검출 없음                    → PASS
  ├─ conf > 해당 클래스 review_max → FAIL (즉시 확정)
  ├─ conf ≥ 해당 클래스 review_min → REVIEW 후보
  └─ 우선순위: FAIL > REVIEW > PASS
     │
     ▼
[DB 저장]  upsert_tile(tile_bgr, verdict, image_path, row, col)
  └─ UNIQUE(image_path, grid_row, grid_col) — 동일 위치 재처리 시 최신값으로 교체
     │
     ▼
[UI 업데이트]
  ├─ GlobalView: 셀 색상 업데이트 + 카메라 위치 표시
  ├─ VisionViewer: 타일 이미지 + 결함 오버레이
  ├─ FilmStrip: 썸네일 추가
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
│  + 그리드 오버레이     │  + 결함 코너 브라켓 오버레이  │
│                       │                             │
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

컬러 PCB 이미지 위에 그리드와 셀 상태를 오버레이합니다.

| 셀 상태 | 테두리 색 | 채우기 색 |
|---------|-----------|-----------|
| UNINSPECTED | 회색 (80,80,80) | 검정 반투명 |
| PASS | 초록 (0,200,0) | 초록 반투명 |
| FAIL | 빨강 (255,50,50) | 빨강 반투명 |
| REVIEW | 노랑 (255,200,0) | 노랑 반투명 |
| 카메라 위치 | Cyan (0,255,255) DashLine | — |

### VisionViewer (`vision_viewer.py`, 읽기전용)

현재 처리 중인 타일을 표시하는 비대화형 뷰어입니다.
- 결함 위치에 **코너 브라켓** 오버레이 (팔 길이 = bbox 너비의 20%)
- 라벨: `{class_name} {conf:.2f}` — 줌 불변 크기 (Consolas 9pt Bold, 검정 70% 불투명 배경)
- 신뢰도 색상: 빨강(FAIL 수준) / 노랑(REVIEW 수준) / 초록(PASS 수준) — 클래스별 REVIEW MIN/MAX(%)에 동적으로 연동

---

## 모델 앙상블 구조

```
weights/
├── best_fold_1.pt  ┐
├── best_fold_2.pt  │
├── best_fold_3.pt  ├─► 5-Fold K-Fold 앙상블 (WBF)
├── best_fold_4.pt  │
└── best_fold_5.pt  ┘
```

- 각 모델이 독립적으로 추론 → `weighted_boxes_fusion()` 으로 박스 병합
- 단순 NMS보다 재현율이 높고 다수결 기반으로 안정적인 검출
- `CROP_PAD_RATIO=0.3` — 결함 주변 30% 패딩으로 썸네일 생성

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
