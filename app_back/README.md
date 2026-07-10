# app_back — REVIEW/FAIL 결함 실시간 수신 · 수동 판정 리뷰 스테이션

`app_front`가 SQLite DB에 기록한 REVIEW/FAIL 결함(defect)을 3초마다 폴링하여 수신하고,
작업자가 키보드 단축키로 각 결함의 최종 판정(Pass/Fail 클래스)을 내립니다.
app_back은 자체 추론을 하지 않습니다 — app_front가 저장 시점에 계산한 bbox/클래스/신뢰도/판정을
그대로 읽어 표시합니다.

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
# 단독 실행 (app_front와 DB를 공유하는 상태에서)
python app_back/run.py

# app_front와 동시 실행 (권장)
# Windows
app_front\run_app.bat
# macOS/Linux
bash app_front/run_app.sh
```

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

> 판정 단축키(Space, 1~6)는 판정 즉시 다음 미검토(pending) 결함으로 자동 이동합니다.

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
  └─► fetch_review_defects(after_id=_last_shown_id) → 새 결함 조회
        └─► 앞쪽 MAX_DEFECTS_PER_POLL_TICK(기본 3)건만 이번 틱에 처리, 나머지는 다음 틱으로 이월
              (REVIEW/FAIL 폭주 시 한 번에 몰아서 처리하면 UI가 멈추는 것을 방지)
        └─► 처리 대상 각 결함에 대해 _process_defect(defect_row):
              ├─ tile_id로 타일 이미지 확보 (캐시 미스 시 DB에서 PNG BLOB 조회 후 디코딩)
              ├─ DefectEntry 생성 (defect_id, tile_id, class_name, confidence, bbox_abs, ai_verdict)
              ├─ bbox 기준 크롭으로 썸네일 생성
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
│      FilmStrip — REVIEW/FAIL 결함 썸네일 (96×96)      │
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

FilmStrip 1건에 대응하는 데이터 클래스 — 결함 1건 = 엔트리 1개.

| 속성 | 타입 | 설명 |
|------|------|------|
| `defect_id` | int | DB defects.id (user_verdict 저장 키) |
| `tile_id` | int | 이 결함이 속한 타일의 DB tiles.id (형제 결함 그룹핑, 타일 이미지 캐시 키) |
| `class_id` / `class_name` | int / str | 결함 클래스 |
| `confidence` | float | 신뢰도 (0.0~1.0) |
| `bbox_abs` | list | 타일 내 절대 좌표 bbox `[x1, y1, x2, y2]` |
| `ai_verdict` | str | app_front가 저장한 AI 판정: REVIEW / FAIL |
| `user_verdict` | str | 작업자 판정: `"pending"` / `"pass"` / `"1"~"6"` |

`MainWindow._tile_cache`(tile_id → 디코딩된 타일 이미지)와 `MainWindow._tile_defects`
(tile_id → 형제 defect_id 목록)가 같은 타일을 공유하는 여러 `DefectEntry`의 이미지 중복
보관을 방지합니다.

### VisionViewer (`vision_viewer.py`, 대화형)

결함 오버레이를 표시하는 대화형 뷰어입니다. 클릭한 결함이 속한 타일의 형제 결함도
함께 표시하되(참고용, non-highlight), 클릭한 결함만 강조 테두리로 표시하고
`zoom_to_detections()`는 그 결함 1건 기준으로 확대합니다.

| 조작 | 동작 |
|------|------|
| 우클릭 드래그 | 패닝 (이미지 이동) |
| 마우스 휠 | 줌 인/아웃 (포인터 중심) |
| `Shift` (Hold) | 강조 표시된 결함(클릭한 것) 1건만 남기고 나머지 숨김 |

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
