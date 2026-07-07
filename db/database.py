"""SQLAlchemy 엔진 및 세션 팩토리 — 단일 진입점."""

import os
from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

_engine = None
_SessionLocal = None


def _get_db_url() -> str:
    """DATABASE_URL 환경변수 → config.yaml → 기본값 순으로 URL 결정."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        url = cfg.get("database", {}).get("url")
        if url:
            return url

    return "postgresql://pcb:pcb1234@localhost:5432/pcb_inspection"


def get_engine():
    global _engine
    if _engine is None:
        url = _get_db_url()
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal()
