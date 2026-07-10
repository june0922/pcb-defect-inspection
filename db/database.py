# SQLite 직접 접근 헬퍼 — 타일 이미지, 판정 결과, 앱 설정을 관리
import json
import sqlite3
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

DB_PATH = Path(__file__).resolve().parent / "inspection.db"

# app_front가 한 번도 실행되지 않아 DB settings가 완전히 비어있는 극단적 상황(예: app_back이
# 먼저 켜진 경우)을 위한 최소 안전망. 진짜 설정 가능한 기본값은 app_front/default_settings.json
# (app_front/defaults_store.py가 소유)이며, db 계층은 그 파일의 존재조차 알지 못한다.
_BOOTSTRAP_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
_BOOTSTRAP_MODEL_PATHS = [f"weights/best_fold_{i}.pt" for i in range(1, 6)]


def _bootstrap_settings() -> dict[str, str]:
    """settings 테이블이 완전히 비어있을 때만 쓰이는 최소 하드코딩 안전망."""
    settings: dict[str, str] = {
        "defect_classes": json.dumps(_BOOTSTRAP_CLASSES),
        "tile_size": "640",
        "overlap_pct": "0",
        "alert_sound": "true",
        "model_paths": json.dumps(_BOOTSTRAP_MODEL_PATHS),
    }
    for cls in _BOOTSTRAP_CLASSES:
        settings[f"review_min_{cls}"] = "30"
        settings[f"review_max_{cls}"] = "70"
    return settings


# 모듈 전역 영구 커넥션 — 최초 호출 시 1회만 생성해 재사용한다. 기존에는 함수 호출마다
# 매번 새 커넥션을 열고 PRAGMA를 재실행해, app_front가 타일을 저장할 때마다(REVIEW 폭주
# 시 초당 여러 건) 이 비용이 누적되어 UI 렉의 핵심 원인이었다. check_same_thread=False로
# 열어 InspectionWorker 백그라운드 스레드와 UI 스레드가 동일 커넥션을 공유하며, 실제 동시
# 접근 안전은 _LOCK이 보장한다. get_db_stats()가 내부적으로 count_by_verdict()를 호출해
# 같은 스레드가 락을 중첩 획득하므로 일반 Lock이 아닌 RLock을 쓴다(그렇지 않으면 데드락).
_CONN: sqlite3.Connection | None = None
_LOCK = threading.RLock()


def _get_connection() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("PRAGMA foreign_keys=ON")
    return _CONN


_TILES_DDL = """
    CREATE TABLE tiles (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        tile_image   BLOB NOT NULL,
        verdict      TEXT NOT NULL,
        image_path   TEXT,
        grid_row     INTEGER,
        grid_col     INTEGER,
        inspected_at TEXT DEFAULT (datetime('now')),
        updated_at   TEXT DEFAULT (datetime('now')),
        UNIQUE(image_path, grid_row, grid_col)
    )
"""

_DEFECTS_DDL = """
    CREATE TABLE IF NOT EXISTS defects (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        tile_id      INTEGER NOT NULL REFERENCES tiles(id) ON DELETE CASCADE,
        class_id     INTEGER NOT NULL,
        class_name   TEXT NOT NULL,
        confidence   REAL NOT NULL,
        bbox_x1      REAL NOT NULL,
        bbox_y1      REAL NOT NULL,
        bbox_x2      REAL NOT NULL,
        bbox_y2      REAL NOT NULL,
        verdict      TEXT NOT NULL,
        user_verdict TEXT,
        created_at   TEXT DEFAULT (datetime('now')),
        updated_at   TEXT DEFAULT (datetime('now'))
    )
"""


def init_db() -> None:
    """DB 초기화 및 스키마 마이그레이션.

    - tiles 테이블: 타일 PNG 이미지 + 위치 메타만 보관 (결함 단위 이전 구버전에는
      user_verdict 컬럼이 있었으나, 판정은 이제 defects 테이블로 이동했다)
    - defects 테이블: 결함 1건 = 1행 (bbox/class/confidence/AI 판정/작업자 판정)
    - settings 테이블: 신규 생성 + 기본값 삽입
    기존 DB가 없으면 처음부터 생성. app_front가 검사 시작마다 clear_all()로 전체
    데이터를 지우는 구조라 구버전 데이터 보존은 하지 않고 재생성한다.
    """
    with _LOCK, _get_connection() as conn:
        # ── tiles 테이블 ──────────────────────────────────────────
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tiles)")}

        if not existing:
            # 최초 생성
            conn.execute(_TILES_DDL)
        elif "user_verdict" in existing:
            # 구버전(결함 단위 이전) 스키마 감지 → 데이터 보존 없이 재생성
            conn.executescript("""
                DROP TABLE IF EXISTS defects;
                DROP TABLE tiles;
            """)
            conn.execute(_TILES_DDL)

        # ── defects 테이블 ────────────────────────────────────────
        conn.execute(_DEFECTS_DDL)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_defects_tile_id ON defects(tile_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_defects_verdict_id ON defects(verdict, id)")

        # ── settings 테이블 ───────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 기본값 삽입 (이미 있는 키는 건드리지 않음 — 안전망 용도)
        defaults = _bootstrap_settings()
        defaults["db_session_id"] = str(uuid.uuid4())

        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (key, value),
            )


# ── tiles 관련 ────────────────────────────────────────────────


def upsert_tile(
    tile_bgr: np.ndarray,
    verdict: str,
    image_path: str,
    grid_row: int,
    grid_col: int,
    detections: list[dict],
) -> None:
    """타일을 DB에 삽입하거나 (같은 위치이면) 최신 결과로 교체.

    detections는 REVIEW/FAIL 등급 결함만 담은 리스트로, 각 원소는
    {class_id, class_name, confidence, bbox_abs, verdict} 키를 가진다.
    같은 위치가 재삽입되면 기존 defects를 전부 지우고 새로 삽입한다.
    """
    _, buf = cv2.imencode(".png", tile_bgr)
    blob = buf.tobytes()
    with _LOCK, _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tiles(tile_image, verdict, image_path, grid_row, grid_col, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(image_path, grid_row, grid_col) DO UPDATE SET
                tile_image   = excluded.tile_image,
                verdict      = excluded.verdict,
                updated_at   = datetime('now')
            """,
            (blob, verdict, image_path, grid_row, grid_col),
        )
        tile_id = conn.execute(
            "SELECT id FROM tiles WHERE image_path=? AND grid_row=? AND grid_col=?",
            (image_path, grid_row, grid_col),
        ).fetchone()[0]

        conn.execute("DELETE FROM defects WHERE tile_id=?", (tile_id,))
        for det in detections:
            x1, y1, x2, y2 = det["bbox_abs"]
            conn.execute(
                """
                INSERT INTO defects(
                    tile_id, class_id, class_name, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_id, det["class_id"], det["class_name"], det["confidence"],
                    x1, y1, x2, y2, det["verdict"],
                ),
            )


def fetch_review_defects(after_id: int = 0) -> list[dict]:
    """verdict가 REVIEW/FAIL인 결함을 after_id 이후 순서로 조회."""
    with _LOCK, _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tile_id, class_id, class_name, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, verdict
            FROM defects
            WHERE verdict IN ('REVIEW', 'FAIL') AND id > ?
            ORDER BY id ASC
            """,
            (after_id,),
        ).fetchall()
    return [
        {
            "id": r[0], "tile_id": r[1], "class_id": r[2], "class_name": r[3],
            "confidence": r[4], "bbox_abs": [r[5], r[6], r[7], r[8]], "verdict": r[9],
        }
        for r in rows
    ]


def get_tile_image(tile_id: int) -> bytes | None:
    """단일 타일의 원본 PNG BLOB 조회. 없으면 None."""
    with _LOCK, _get_connection() as conn:
        row = conn.execute(
            "SELECT tile_image FROM tiles WHERE id=?",
            (tile_id,),
        ).fetchone()
    return row[0] if row else None


def count_by_verdict() -> dict:
    """판정별 타일 수 집계."""
    with _LOCK, _get_connection() as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) FROM tiles GROUP BY verdict"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def save_defect_verdict(defect_id: int, user_verdict: str | None) -> None:
    """app_back 작업자 판정을 defects 테이블에 저장 (결함 1건 단위)."""
    with _LOCK, _get_connection() as conn:
        conn.execute(
            "UPDATE defects SET user_verdict=?, updated_at=datetime('now') WHERE id=?",
            (user_verdict, defect_id),
        )


def clear_all() -> None:
    """모든 타일/결함 삭제 + AUTO_INCREMENT 리셋 + db_session_id 갱신."""
    new_session = str(uuid.uuid4())
    with _LOCK, _get_connection() as conn:
        conn.execute("DELETE FROM defects")
        conn.execute("DELETE FROM tiles")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('tiles', 'defects')")
        conn.execute(
            "UPDATE settings SET value=?, updated_at=datetime('now') WHERE key='db_session_id'",
            (new_session,),
        )


def get_db_stats() -> dict:
    """타일 수(판정별) + DB 파일 크기(WAL/SHM 포함) 반환."""
    stats = count_by_verdict()
    stats["_total"] = sum(stats.values())

    total_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    for suffix in ("-wal", "-shm"):
        wal_path = DB_PATH.with_name(DB_PATH.name + suffix)
        if wal_path.exists():
            total_bytes += wal_path.stat().st_size
    stats["_db_bytes"] = total_bytes
    return stats


# ── settings 관련 ─────────────────────────────────────────────


def get_settings() -> dict:
    """settings 테이블 전체를 {key: value} dict로 반환."""
    with _LOCK, _get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r[0]: r[1] for r in rows}


def update_setting(key: str, value: str) -> None:
    """단일 설정 값 업데이트."""
    with _LOCK, _get_connection() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
            (key, value),
        )


def update_settings(settings: dict) -> None:
    """여러 설정 값 일괄 업데이트."""
    with _LOCK, _get_connection() as conn:
        for key, value in settings.items():
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
                (key, str(value)),
            )
