# app_back — REVIEW 결함 실시간 수신 · 수동 판정 리뷰 스테이션

`app_front`가 SQLite DB에 기록한 REVIEW 결함(defect)을 3초마다 폴링하여 필름스트립에 수신하고,
작업자가 키보드 단축키로 각 결함의 최종 판정(Pass/Fail 클래스)을 내립니다. 필름스트립에는
사람이 검토해야 할 REVIEW만 표시되며, FAIL(확정 결함)은 올라오지 않습니다 — 다만 우측
LocalView에서는 같은 타일의 FAIL 형제 결함도 참고용으로 함께 표시됩니다.
app_back은 자체 추론을 하지 않습니다 — app_front가 저장 시점에 계산한 bbox/클래스/신뢰도/판정을
그대로 읽어 표시합니다. LocalView에서 결함 bbox를 드래그로 이동/리사이즈할 수 있으며, 결과는
즉시 DB에 저장되어 재시작해도 유지됩니다.

---

## 파일 구성

```
app_back/
├── run.py               # 진입점 (다크 테마 적용 후 MainWindow 실행)
├── main_ui.py           # 메인 윈도우 (MainWindow, DefectEntry)
└── vision_viewer.py     # 결함 오버레이 대화형 뷰어 (VisionViewer)
```

---

## 실행 방법

```bash
# Windows
app_back\run_app.bat
# macOS/Linux
bash app_back/run_app.sh

# 또는 직접 실행
python app_back/run.py
```

app_front와 app_back은 각자 독립적으로 실행되는 별개의 프로그램입니다 — 어느 쪽을 먼저
켜도, 하나만 켜도, 둘 다 켜져 있는 상태에서 하나를 재시작해도 DB(`db/database.py`)를
통해 정상적으로 연동됩니다. `init_db()`는 두 프로세스가 각자 호출해도 안전합니다
(`CREATE TABLE IF NOT EXISTS` + `INSERT OR IGNORE`만 사용).

> app_back은 YOLO 모델을 로딩하지 않으므로 앱 시작 즉시 DB 폴링이 시작됩니다.

---

## 단축키

| 키 | 동작 |
|----|------|
| `Space` | 현재 결함 → **Pass** (정상/오탐 판정) |
| `1` ~ `6` | 현재 결함 → **Fail** (결함 클래스 번호: 1=open, 2=short, 3=mousebite, 4=spur, 5=copper, 6=pinhole) |
| `←` | 이전 결함으로 이동 |
| `→` | 다음 결함으로 이동 |
| `W` / `S` | VisionViewer 위/아래 패닝 |
| `A` / `D` | VisionViewer 좌/우 패닝 |
| `Q` / `E` | VisionViewer 축소 / 확대 |
| `Shift` (Hold) | 현재 선택된 결함 오버레이 외 나머지 숨김 |
| `Ctrl` | 아직 판정하지 않은 가장 앞쪽 결함으로 포커스 이동 |
| `ESC` | 현재 결함의 판정을 취소하고 미검토 상태로 복원 |
| 좌클릭 드래그 (VisionViewer) | 강조된 결함의 bbox 이동(내부 클릭) 또는 리사이즈(핸들 클릭) |

> 판정 단축키(Space, 1~6)는 판정 즉시 다음 미검토(pending) 결함으로 자동 이동합니다.
> bbox 드래그는 Shift 상태와 무관하게 항상 동일하게 동작합니다 — Shift는 오버레이 숨김
> 토글 전용이며 편집 로직과는 완전히 분리되어 있습니다.

---

## DB 폴링 메커니즘

```
앱 시작
  └─► init_db() + 초기 설정 로드 → QTimer(3000ms) 시작 (모델 로딩 대기 없음)

QTimer 3초마다 _poll_db() 호출
  │
  ├─► get_settings() → DB에서 현재 설정 읽기
  │     ├─ db_session_id 변경 감지?
  │     │     └─► _on_db_reset(): filmstrip/타일 캐시 전체 초기화,
  │     │                         _last_shown_id = 0
  │     │
  │     └─ REVIEW 밴드 변경 감지?
  │           └─► per_class_bands 갱신 (색상 판정에만 사용 — 결함의 REVIEW/FAIL
  │               여부 자체는 DB에 저장된 verdict를 그대로 신뢰하고 재판정하지 않음)
  │
  └─► fetch_review_defects(after_id=_last_shown_id) → 새 REVIEW 결함만 조회 (FAIL은 제외)
        └─► 앞쪽 MAX_DEFECTS_PER_POLL_TICK(기본 3)건만 이번 틱에 처리, 나머지는 다음 틱으로 이월
              (REVIEW 폭주 시 한 번에 몰아서 처리하면 UI가 멈추는 것을 방지)
        └─► 처리 대상 각 결함에 대해 _process_defect(defect_row):
              ├─ tile_id로 타일 이미지 확보 (캐시 미스 시 DB에서 PNG BLOB 조회 후 디코딩)
              ├─ DefectEntry 생성 (defect_id, tile_id, class_name, confidence, bbox_abs, ai_verdict,
              │  user_verdict — DB에 저장된 기존 판정이 있으면 재시작 후에도 그대로 복원)
              ├─ bbox 기준 크롭으로 썸네일 생성 (복원된 판정에 맞는 테두리색/배지도 함께 반영)
              └─ FilmStrip 썸네일 추가 (개수 상한 없음 — 미판정 대기열이라 의도적으로 무제한 누적)
```

---

## 판정 흐름

```
작업자 키 입력 (Space / 1~6)
  └─► _verdict_current(verdict)
        ├─ 디바운스 (100ms 이내 중복 입력 무시)
        ├─ entry.user_verdict = verdict
        ├─ save_defect_verdict(defect_id, verdict) → DB 저장
        │     (defects.user_verdict 컬럼 업데이트 — 이 결함 1건만, 같은 타일의
        │      다른 결함은 독립적으로 판정됨)
        ├─ 썸네일 테두리 색 갱신
        │     pending → 회색
        │     pass    → 초록
        │     1~6     → 빨강 + 번호 배지
        └─ 다음 pending 결함으로 자동 이동
```

## bbox 편집 흐름

```
VisionViewer에서 강조(highlight)된 결함의 핸들/내부를 좌클릭 드래그
  ├─ mousePressEvent: 히트테스트로 편집 모드 결정 (move | tl/tm/tr/ml/mr/bl/bm/br)
  │     (핸들은 강조된 결함에 대해서만 존재하므로, 형제/비강조 결함은 편집 불가)
  ├─ mouseMoveEvent: bbox 재계산 + 브라켓/라벨/핸들 경량 갱신 (DB 접근 없음)
  │     최소 크기(10px) 클램프, 반대쪽을 넘어서는 반전은 불가
  └─ mouseReleaseEvent: bbox_edited 시그널 emit
        └─► MainWindow._on_bbox_edited(new_bbox_abs)
              ├─ entry.bbox_abs 갱신 (메모리)
              ├─ update_defect_bbox(defect_id, new_bbox_abs) → DB 저장 (드래그 종료 시 1회)
              └─ FilmStrip 썸네일 갱신 (bbox가 바뀌었으므로 크롭 영역도 다시 계산)
```

---

## UI 레이아웃

```
┌─────────────────────────────────────────────────────┐
├───────────────────────┬─────────────────────────────┤
│   GlobalView (30%)    │    VisionViewer (70%)        │
│                       │                             │
│  결함이 속한 타일 전체 │  결함 확대 뷰 (대화형)        │
│  미리보기(읽기전용)    │  + 코너 브라켓 오버레이       │
│                       │                             │
├───────────────────────┴─────────────────────────────┤
│         FilmStrip — REVIEW 결함 썸네일 (96×96)         │
│         가로 스크롤, 판정 상태별 테두리 색             │
├─────────────────────────────────────────────────────┤
│ StatusBar: [N/total] Pass: N | Fail: N | 미검토: N   │
└─────────────────────────────────────────────────────┘
```

같은 타일에 속한 여러 결함(형제 엔트리)을 필름스트립에서 각각 클릭하면, GlobalView는
매번 동일한 타일 이미지를 보여주고 LocalView는 클릭한 결함 위치로 확대됩니다.

---

## 주요 컴포넌트

### DefectEntry

FilmStrip 1건에 대응하는 데이터 클래스 — REVIEW 결함 1건 = 엔트리 1개.

| 속성 | 타입 | 설명 |
|------|------|------|
| `defect_id` | int | DB defects.id (user_verdict/bbox 저장 키) |
| `tile_id` | int | 이 결함이 속한 타일의 DB tiles.id (형제 결함 그룹핑, 타일 이미지 캐시 키) |
| `class_id` / `class_name` | int / str | 결함 클래스 |
| `confidence` | float | 신뢰도 (0.0~1.0) |
| `bbox_abs` | list | 타일 내 절대 좌표 bbox `[x1, y1, x2, y2]` (드래그 편집으로 갱신됨) |
| `ai_verdict` | str | app_front가 저장한 AI 판정: 필름스트립 엔트리는 항상 REVIEW |
| `user_verdict` | str | 작업자 판정: `"pending"` / `"pass"` / `"1"~"6"` (DB에서 복원되어 재시작해도 유지) |

`MainWindow._tile_cache`(tile_id → 디코딩된 타일 이미지)와 `MainWindow._tile_defects`
(tile_id → 형제 defect_id 목록)가 같은 타일을 공유하는 여러 `DefectEntry`의 이미지 중복
보관을 방지합니다. LocalView에 형제(FAIL 포함) 결함을 그릴 때는 이 캐시가 아니라
`fetch_tile_defects(tile_id)`로 DB에서 직접 REVIEW+FAIL 전체를 조회합니다.

### VisionViewer (`vision_viewer.py`, 대화형)

결함 오버레이를 표시하는 대화형 뷰어입니다. 클릭한 결함이 속한 타일의 형제 결함도
함께 표시하되(참고용, non-highlight), 클릭한 결함만 강조 테두리 + 편집 핸들로 표시하고
`zoom_to_detections()`는 그 결함 1건 기준으로 확대합니다.

| 조작 | 동작 |
|------|------|
| 우클릭 드래그 | 패닝 (이미지 이동) |
| 마우스 휠 | 줌 인/아웃 (포인터 중심) |
| 좌클릭 드래그 (강조 결함 내부) | bbox 이동 |
| 좌클릭 드래그 (강조 결함의 8방향 핸들) | bbox 리사이즈 |
| `Shift` (Hold) | 강조 표시된 결함(클릭한 것) 1건만 남기고 나머지 숨김 |

강조(highlight)된 결함에만 이동용 히트 영역과 8방향(코너 4 + 변 중점 4) 리사이즈 핸들이
생성됩니다 — 형제(non-highlight) 결함은 핸들 자체가 없으므로 실수로 클릭해도 편집되지
않습니다. 핸들은 `ItemIgnoresTransformations`로 줌 레벨과 무관하게 화면상 고정 크기(9px)를
유지합니다. 드래그 중에는 브라켓/라벨/핸들만 경량으로 다시 그리며(DB 접근 없음), 드래그가
끝나는 순간(`mouseReleaseEvent`) `bbox_edited` 시그널로 최종 좌표를 한 번만 전달해
`MainWindow`가 DB에 저장합니다.

결함 박스 색상은 `confidence_color(conf, class_name, per_class_bands)` 함수로 per-class
판단하며, `per_class_bands`는 3초 폴링마다 DB(app_front의 Option)에서 갱신됩니다:
- **빨강**: conf > class review_max (FAIL 수준)
- **노랑**: conf ≥ class review_min (REVIEW 수준)

단, 결함이 REVIEW/FAIL 등급인지 자체는 DB에 저장된 `defects.verdict`(app_front가 저장
시점에 판정한 값)를 그대로 신뢰합니다 — app_back은 재추론하지 않으므로 밴드 변경 후에도
이미 표시된 결함이 사라지거나 재필터링되지 않습니다.

---

## 의존성

```
PyQt5 ≥ 5.15
opencv-python
numpy
```
