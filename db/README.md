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

타일별 AI 판정 결과와 작업자 최종 판정을 저장합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                            tiles                                     │
├──────────────┬───────────┬──────────────────────────────────────────┤
│ 컬럼         │ 타입      │ 설명                                       │
├──────────────┼───────────┼──────────────────────────────────────────┤
│ id           │ INTEGER   │ PK, AUTOINCREMENT                         │
│ tile_image   │ BLOB      │ PNG 인코딩된 640×640 타일 이미지           │
│ verdict      │ TEXT      │ AI 판정: PASS / FAIL / REVIEW             │
│ image_path   │ TEXT      │ 원본 이미지 파일 절대 경로                 │
│ grid_row     │ INTEGER   │ 타일 그리드 행 번호 (0-based)             │
│ grid_col     │ INTEGER   │ 타일 그리드 열 번호 (0-based)             │
│ user_verdict │ TEXT      │ 작업자 판정: pass / 1~6 / NULL(미판정)    │
│ inspected_at │ TEXT      │ 검사 시각 (datetime, 자동 삽입)            │
│ updated_at   │ TEXT      │ 최종 갱신 시각 (datetime, 자동 삽입)       │
└──────────────┴───────────┴──────────────────────────────────────────┘
UNIQUE(image_path, grid_row, grid_col)
```

**UNIQUE 제약 & Upsert 동작**  
동일한 `(image_path, grid_row, grid_col)` 조합이 다시 삽입되면 기존 행을 최신 결과로 교체합니다.  
재처리(같은 폴더 재오픈) 시 중복 없이 `tile_image`, `verdict`, `updated_at`만 갱신하고 `user_verdict`는 `NULL`로 초기화됩니다.

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

**기본값 (15개 키, 최초 실행 시 자동 삽입)**

| 키 | 기본값 | 설명 |
|----|--------|------|
| `iou_threshold` | `"0.45"` | WBF / NMS IoU 임계값 |
| `alert_sound` | `"true"` | FAIL 검출 시 경고음 활성화 |
| `review_min_open` | `"30"` | open 결함 REVIEW 하한 (%) |
| `review_max_open` | `"70"` | open 결함 REVIEW 상한, 초과 시 FAIL (%) |
| `review_min_short` | `"30"` | short 결함 REVIEW 하한 |
| `review_max_short` | `"70"` | short 결함 REVIEW 상한 |
| `review_min_mousebite` | `"30"` | mousebite 결함 REVIEW 하한 |
| `review_max_mousebite` | `"70"` | mousebite 결함 REVIEW 상한 |
| `review_min_spur` | `"30"` | spur 결함 REVIEW 하한 |
| `review_max_spur` | `"70"` | spur 결함 REVIEW 상한 |
| `review_min_copper` | `"30"` | copper 결함 REVIEW 하한 |
| `review_max_copper` | `"70"` | copper 결함 REVIEW 상한 |
| `review_min_pinhole` | `"30"` | pinhole 결함 REVIEW 하한 |
| `review_max_pinhole` | `"70"` | pinhole 결함 REVIEW 상한 |
| `db_session_id` | UUID4 | DB 초기화 감지용 세션 토큰 |

---

## DB 세션 리셋 메커니즘

```
app_front: clear_all() 호출
    └─► tiles 전체 삭제
    └─► sqlite_sequence 리셋 (AUTOINCREMENT 1부터 재시작)
    └─► db_session_id = 새 UUID4  ◄── 핵심 신호

app_back: _poll_db() (3초마다)
    └─► get_settings()["db_session_id"] 조회
    └─► 이전 값과 다르면? → _on_db_reset() 실행
            └─► filmstrip 전체 초기화
            └─► _last_shown_id = 0
            └─► 이후 폴링부터 새 타일 수신 재시작
```

---

## API 함수 목록

| 함수 | 시그니처 | 역할 |
|------|----------|------|
| `init_db` | `() → None` | 테이블 생성 또는 구버전 스키마 자동 마이그레이션, 기본값 삽입 |
| `upsert_tile` | `(tile_bgr, verdict, image_path, grid_row, grid_col) → None` | 타일 삽입/교체 (동일 위치면 최신값으로 덮어씀) |
| `fetch_review_tiles` | `(after_id=0) → list[dict]` | `verdict='REVIEW'`인 타일을 `after_id` 이후 ID순으로 반환 |
| `count_by_verdict` | `() → dict` | `{verdict: count}` 형태로 판정별 타일 수 집계 |
| `save_user_verdict` | `(tile_id, user_verdict) → None` | 작업자 판정 저장 (`user_verdict` + `updated_at` 업데이트) |
| `clear_all` | `() → None` | tiles 전체 삭제 + AUTOINCREMENT 리셋 + db_session_id 갱신 |
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

구버전 DB(`image_path` 컬럼 없음)가 감지되면 `init_db()` 호출 시 자동으로 테이블을 재생성합니다.  
기존 `id`, `tile_image`, `verdict`, `inspected_at` 데이터는 보존됩니다.

```
PRAGMA table_info(tiles) → image_path 컬럼 없음?
    └─► CREATE TABLE tiles_new (신규 스키마)
    └─► INSERT INTO tiles_new SELECT FROM tiles (기존 데이터 복사)
    └─► DROP TABLE tiles
    └─► ALTER TABLE tiles_new RENAME TO tiles
```
