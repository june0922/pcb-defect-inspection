# SQLite 직접 접근 헬퍼 — 타일 이미지와 판정 결과를 단일 테이블에 저장

import sqlite3
import cv2
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "inspection.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """tiles 테이블 생성 (최초 1회 또는 idempotent)."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_image   BLOB NOT NULL,
                verdict      TEXT NOT NULL,
                inspected_at TEXT DEFAULT (datetime('now'))
            )
        """)


def insert_tile(tile_bgr: np.ndarray, verdict: str) -> None:
    """640×640 BGR 타일 이미지를 PNG BLOB으로 인코딩하여 DB에 삽입."""
    _, buf = cv2.imencode(".png", tile_bgr)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tiles (tile_image, verdict) VALUES (?, ?)",
            (buf.tobytes(), verdict),
        )


def fetch_review_tiles(after_id: int = 0) -> list[dict]:
    """verdict='REVIEW' 인 타일을 after_id 이후부터 오름차순으로 조회."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, tile_image FROM tiles WHERE verdict='REVIEW' AND id > ? ORDER BY id ASC",
            (after_id,),
        ).fetchall()
    return [{"id": r[0], "tile_image": r[1]} for r in rows]


def count_by_verdict() -> dict:
    """판정별 타일 수 반환 (상태 표시용)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) FROM tiles GROUP BY verdict"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def clear_all() -> None:
    """tiles 테이블 전체 삭제 (새 검사 세션 시작 전 호출)."""
    with _connect() as conn:
        conn.execute("DELETE FROM tiles")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='tiles'")
