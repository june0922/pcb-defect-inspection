"""DB 초기화 — 테이블 생성 (없을 때만).

실행:
    python -m db.init_db
"""

from .database import get_engine
from .models import Base


def init_db() -> None:
    """모든 테이블을 create_all로 생성 (이미 존재하면 무시)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"[DB] 초기화 완료 → {engine.url}")


if __name__ == "__main__":
    init_db()
