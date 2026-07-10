# db — SQLite 단일 진실 공급원 (SSOT)

PCB 결함 검사 시스템의 모든 데이터(타일 이미지, AI 판정, 작업자 판정, 앱 설정)를 단일 SQLite 파일로 관리합니다.  
`app_front`(쓰기)와 `app_back`(읽기)이 동시에 접근하므로 **WAL(Write-Ahead Logging) 모드**를 사용합니다.

---

## 파일 구성

```
db/
├── database.py      # DB 헬퍼 함수 모음
└── inspection.db    # SQLite DB 파일 (앱 최초 실행 시 자동 생성)
```

---

## DB 스키마

### tiles 테이블

타일 PNG 이미지와 위치 메타만 보관합니다. 결함별 판정은 아래 `defects` 테이블로 분리되어 있습니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                            tiles                                     │
├──────────────┬───────────┬──────────────────────────────────────────┤
│ 컬럼         │ 타입      │ 설명                                       │
├──────────────┼───────────┼──────────────────────────────────────────┤
│ id           │ INTEGER   │ PK, AUTOINCREMENT                         │
│ tile_image   │ BLOB      │ PNG 인코딩된 타일 이미지 (크기는 타일크기 설정에 따름)│
│ verdict      │ TEXT      │ 타일 전체 AI 판정 요약: PASS / FAIL / REVIEW│
│ image_path   │ TEXT      │ 원본 이미지 파일 절대 경로                 │
│ grid_row     │ INTEGER   │ 타일 그리드 행 번호 (0-based)             │
│ grid_col     │ INTEGER   │ 타일 그리드 열 번호 (0-based)             │
│ inspected_at │ TEXT      │ 검사 시각 (datetime, 자동 삽입)            │
│ updated_at   │ TEXT      │ 최종 갱신 시각 (datetime, 자동 삽입)       │
└──────────────┴───────────┴──────────────────────────────────────────┘
UNIQUE(image_path, grid_row, grid_col)
```

**UNIQUE 제약 & Upsert 동작**  
동일한 `(image_path, grid_row, grid_col)` 조합이 다시 삽입되면 기존 행을 최신 결과로 교체합니다.  
재처리(같은 폴더 재오픈) 시 `tile_image`, `verdict`, `updated_at`이 갱신되고, 그 타일에 속한 기존 `defects` 행은 전부 삭제된 뒤 새 결함으로 재삽입됩니다.

### defects 테이블

결함(detection) 1건 = 1행. app_front가 계산한 bbox/클래스/신뢰도/AI 판정을 그대로 저장하며, app_back은 이 값을 재추론 없이 그대로 읽어 표시합니다. **REVIEW/FAIL 등급만 저장되며 PASS 등급은 저장되지 않습니다** (app_back이 PASS를 쓰지 않으므로 행 수를 절약).

```
┌─────────────────────────────────────────────────────────────────────┐
│                            defects                                   │
├──────────────┬───────────┬──────────────────────────────────────────┤
│ 컬럼         │ 타입      │ 설명                                       │
├──────────────┼───────────┼──────────────────────────────────────────┤
│ id           │ INTEGER   │ PK, AUTOINCREMENT                         │
│ tile_id      │ INTEGER   │ FK → tiles(id), ON DELETE CASCADE         │
│ class_id     │ INTEGER   │ 모델 클래스 정수 ID                        │
│ class_name   │ TEXT      │ 결함 클래스명                              │
│ confidence   │ REAL      │ 신뢰도 (0.0~1.0)                          │
│ bbox_x1~y2   │ REAL      │ 타일 내 절대 좌표 bbox                     │
│ verdict      │ TEXT      │ 결함 1건의 AI 판정: REVIEW / FAIL          │
│ user_verdict │ TEXT      │ 작업자 판정: pass / 1~6 / NULL(미판정)    │
│ created_at   │ TEXT      │ 생성 시각 (datetime, 자동 삽입)            │
│ updated_at   │ TEXT      │ 최종 갱신 시각 (datetime, 자동 삽입)       │
└──────────────┴───────────┴──────────────────────────────────────────┘
INDEX(tile_id), INDEX(verdict, id)
```

app_back의 필름스트립 엔트리는 이 테이블의 행 1개와 1:1 대응합니다. 같은 `tile_id`를 공유하는 여러 결함(형제 결함)은 서로 독립적으로 작업자 판정이 매겨집니다.

---

### settings 테이블

앱 설정을 키-값 쌍으로 저장합니다. app_front가 쓰고, app_back이 3초마다 읽어 동기화합니다.

```
┌──────────────────────────────────────────────────┐
│                    settings                       │
├────────────┬──────────┬──────────────────────────┤
│ 컬럼       │ 타입     │ 설명                       │
├────────────┼──────────┼──────────────────────────┤
│ key        │ TEXT     │ PK — 설정 키               │
│ value      │ TEXT     │ 설정 값 (문자열)            │
│ updated_at │ TEXT     │ 갱신 시각 (datetime)        │
└────────────┴──────────┴──────────────────────────┘
```

**기본값 키**

클래스별 `review_min_*`/`review_max_*` 쌍이 있어 실제 키 개수는 등록된 결함 클래스 수에 따라 달라진다(고정 개수가 아니다). 아래는 `defect_classes`가 기본 6개일 때의 예시다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `defect_classes` | `'["open","short",...]'` (JSON 배열) | 활성 결함 클래스 목록 |
| `tile_size` | `"640"` | 타일 한 변의 길이(px) |
| `overlap_pct` | `"0"` | 인접 타일 간 겹침 비율(%) |
| `alert_sound` | `"true"` | FAIL 검출 시 경고음 활성화 |
| `model_paths` | `'["...best_fold_1.pt", ...]'` (JSON 배열) | 검사 모델(.pt) 경로 목록, 1개 이상 |
| `review_min_{class}` | `"30"` | 클래스별 REVIEW 하한 (%), 클래스마다 1개씩 |
| `review_max_{class}` | `"70"` | 클래스별 REVIEW 상한, 초과 시 FAIL (%), 클래스마다 1개씩 |
| `db_session_id` | UUID4 | DB 초기화 감지용 세션 토큰 |

**진짜 기본값은 `app_front/default_settings.json`이 소유한다.** 이 DB 계층(`db/database.py`의 `_bootstrap_settings()`)은 위 표와 동일한 형태의 **최소 하드코딩 안전망**만 갖고 있으며, `init_db()`가 `INSERT OR IGNORE`로 시딩하므로 이미 값이 있으면 절대 덮어쓰지 않는다. 이 안전망은 app_back이 app_front보다 먼저 실행되어 DB가 완전히 비어있는 극단적 상황에서만 의미가 있다 — 정상적인 흐름에서는 app_front가 시작할 때마다 `default_settings.json`의 실제 값으로 강제 초기화한다.

안전망의 `model_paths` 기본값(`weights/best_fold_1~5.pt`)은 학습 파이프라인 폴더를 가리키며, `app_front/default_settings.json`의 실제 배포 기본값(`app_front/models/best_fold_1~5.pt`)과는 **다른 경로**다 — 안전망은 최후의 수단일 뿐 정상 배포 경로와 일치할 필요가 없다.

---

## DB 세션 리셋 메커니즘

```
app_front: clear_all() 호출
    └─► defects 전체 삭제 → tiles 전체 삭제
    └─► sqlite_sequence 리셋 (tiles/defects 둘 다 AUTOINCREMENT 1부터 재시작)
    └─► db_session_id = 새 UUID4  ◄── 핵심 신호

app_back: _poll_db() (3초마다)
    └─► get_settings()["db_session_id"] 조회
    └─► 이전 값과 다르면? → _on_db_reset() 실행
            └─► filmstrip 전체 초기화 (_tile_cache/_tile_defects도 함께 초기화)
            └─► _last_shown_id = 0
            └─► 이후 폴링부터 새 결함 수신 재시작
```

---

## API 함수 목록

| 함수 | 시그니처 | 역할 |
|------|----------|------|
| `init_db` | `() → None` | 테이블 생성 또는 구버전 스키마 자동 마이그레이션, 기본값 삽입 |
| `upsert_tile` | `(tile_bgr, verdict, image_path, grid_row, grid_col, detections) → None` | 타일 삽입/교체 + 그 타일의 defects를 전부 지우고 재삽입 (REVIEW/FAIL 등급만) |
| `fetch_review_defects` | `(after_id=0) → list[dict]` | `verdict IN ('REVIEW','FAIL')`인 결함을 `after_id` 이후 ID순으로 반환 (tile_id 포함) |
| `get_tile_image` | `(tile_id) → bytes or None` | 단일 타일의 원본 PNG BLOB 조회 |
| `count_by_verdict` | `() → dict` | `{verdict: count}` 형태로 판정별 타일 수 집계 |
| `save_defect_verdict` | `(defect_id, user_verdict) → None` | 결함 1건의 작업자 판정 저장 (`user_verdict` + `updated_at` 업데이트) |
| `clear_all` | `() → None` | defects/tiles 전체 삭제 + AUTOINCREMENT 리셋 + db_session_id 갱신 |
| `get_db_stats` | `() → dict` | 판정별 카운트 + `_total` + `_db_bytes` (파일 크기) |
| `get_settings` | `() → dict` | settings 테이블 전체를 `{key: value}` dict로 반환 |
| `update_setting` | `(key, value) → None` | 단일 설정 키 UPSERT |
| `update_settings` | `(settings: dict) → None` | 여러 설정 키 일괄 UPSERT |

---

## WAL 모드 설명

기본 SQLite 저널 모드(DELETE)는 쓰기 시 전체 DB를 잠가 동시 읽기를 차단합니다.  
WAL 모드에서는 쓰기가 별도 WAL 파일에 기록되므로 **app_front 쓰기와 app_back 읽기가 동시에 가능**합니다.

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
```

---

## 스키마 마이그레이션

구버전 DB(결함 단위 이전, `tiles.user_verdict` 컬럼 존재)가 감지되면 `init_db()` 호출 시
`defects`/`tiles`를 DROP 후 신규 스키마로 재생성합니다. app_front가 검사 시작마다
`clear_all()`로 전체 데이터를 지우는 구조라 구버전 데이터 보존은 하지 않습니다.

```
PRAGMA table_info(tiles) → user_verdict 컬럼 존재?
    └─► DROP TABLE IF EXISTS defects
    └─► DROP TABLE tiles
    └─► CREATE TABLE tiles (신규 스키마)
    └─► CREATE TABLE IF NOT EXISTS defects (신규 스키마)
```
