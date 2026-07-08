# SQLite 직접 접근 헬퍼 — 타일 이미지, 판정 결과, 앱 설정을 관리
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np

DB_PATH = Path(__file__).resolve().parent / "inspection.db"

# 6개 결함 클래스 이름 (data.yaml 기준)
_DEFECT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]

# settings 테이블 기본값
_DEFAULT_SETTINGS: dict[str, str] = {
    "iou_threshold": "0.45",
    "alert_sound": "true",
    **{f"review_min_{cls}": "30" for cls in _DEFECT_CLASSES},
    **{f"review_max_{cls}": "70" for cls in _DEFECT_CLASSES},
    "db_session_id": "",  # init_db()에서 uuid4로 채움
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """DB 초기화 및 스키마 마이그레이션.

    - tiles 테이블: image_path/grid_row/grid_col/user_verdict/updated_at 컬럼 추가
    - settings 테이블: 신규 생성 + 기본값 삽입
    기존 DB가 없으면 처음부터 생성.
    """
    with _connect() as conn:
        # ── tiles 테이블 ──────────────────────────────────────────
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tiles)")}

        if not existing:
            # 최초 생성
            conn.execute("""
                CREATE TABLE tiles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_image   BLOB NOT NULL,
                    verdict      TEXT NOT NULL,
                    image_path   TEXT,
                    grid_row     INTEGER,
                    grid_col     INTEGER,
                    user_verdict TEXT,
                    inspected_at TEXT DEFAULT (datetime('now')),
                    updated_at   TEXT DEFAULT (datetime('now')),
                    UNIQUE(image_path, grid_row, grid_col)
                )
            """)
        elif "image_path" not in existing:
            # 구버전 스키마 마이그레이션: 테이블 재생성 (UNIQUE 제약 추가)
            conn.executescript("""
                CREATE TABLE tiles_new (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_image   BLOB NOT NULL,
                    verdict      TEXT NOT NULL,
                    image_path   TEXT,
                    grid_row     INTEGER,
                    grid_col     INTEGER,
                    user_verdict TEXT,
                    inspected_at TEXT DEFAULT (datetime('now')),
                    updated_at   TEXT DEFAULT (datetime('now')),
                    UNIQUE(image_path, grid_row, grid_col)
                );
                INSERT INTO tiles_new(id, tile_image, verdict, inspected_at)
                    SELECT id, tile_image, verdict, inspected_at FROM tiles;
                DROP TABLE tiles;
                ALTER TABLE tiles_new RENAME TO tiles;
            """)

        # ── settings 테이블 ───────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 기본값 삽입 (이미 있는 키는 건드리지 않음)
        defaults = dict(_DEFAULT_SETTINGS)
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
) -> None:
    """타일을 DB에 삽입하거나 (같은 위치이면) 최신 결과로 교체."""
    _, buf = cv2.imencode(".png", tile_bgr)
    blob = buf.tobytes()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tiles(tile_image, verdict, image_path, grid_row, grid_col, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(image_path, grid_row, grid_col) DO UPDATE SET
                tile_image   = excluded.tile_image,
                verdict      = excluded.verdict,
                user_verdict = NULL,
                updated_at   = datetime('now')
            """,
            (blob, verdict, image_path, grid_row, grid_col),
        )


def fetch_review_tiles(after_id: int = 0) -> list[dict]:
    """verdict='REVIEW'인 타일을 after_id 이후 순서로 조회."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, tile_image FROM tiles WHERE verdict='REVIEW' AND id > ? ORDER BY id ASC",
            (after_id,),
        ).fetchall()
    return [{"id": r[0], "tile_image": r[1]} for r in rows]


def get_tile_image(tile_id: int) -> bytes | None:
    """단일 타일의 원본 PNG BLOB 조회. 없으면 None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT tile_image FROM tiles WHERE id=?",
            (tile_id,),
        ).fetchone()
    return row[0] if row else None


def count_by_verdict() -> dict:
    """판정별 타일 수 집계."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) FROM tiles GROUP BY verdict"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def save_user_verdict(tile_id: int, user_verdict: str) -> None:
    """app_back 작업자 판정을 tiles 테이블에 저장."""
    with _connect() as conn:
        conn.execute(
            "UPDATE tiles SET user_verdict=?, updated_at=datetime('now') WHERE id=?",
            (user_verdict, tile_id),
        )


def clear_all() -> None:
    """모든 타일 삭제 + AUTO_INCREMENT 리셋 + db_session_id 갱신."""
    new_session = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute("DELETE FROM tiles")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='tiles'")
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
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r[0]: r[1] for r in rows}


def update_setting(key: str, value: str) -> None:
    """단일 설정 값 업데이트."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
            (key, value),
        )


def update_settings(settings: dict) -> None:
    """여러 설정 값 일괄 업데이트."""
    with _connect() as conn:
        for key, value in settings.items():
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
                (key, str(value)),
            )
