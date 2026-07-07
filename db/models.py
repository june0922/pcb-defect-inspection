"""SQLAlchemy ORM 모델 — InspectionSession → Board → TileInspection → Detection / Review."""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class InspectionSession(Base):
    """검사 세션 1건 (app_front·app_back·web_hwang 공용)."""

    __tablename__ = "inspection_sessions"

    session_id      = Column(String, primary_key=True, default=_uuid)
    source          = Column(String, nullable=False)          # 'app_front'|'app_back'|'web_hwang'
    started_at      = Column(DateTime, default=datetime.utcnow)
    ended_at        = Column(DateTime, nullable=True)
    target_folder   = Column(String, nullable=True)
    pass_threshold  = Column(Float, default=0.3)
    fail_threshold  = Column(Float, default=0.7)
    iou_threshold   = Column(Float, default=0.45)
    total_tiles     = Column(Integer, default=0)
    pass_count      = Column(Integer, default=0)
    fail_count      = Column(Integer, default=0)
    review_count    = Column(Integer, default=0)
    fpy             = Column(Float, default=0.0)              # First Pass Yield (%)
    avg_inference_ms = Column(Float, default=0.0)
    throughput      = Column(Float, default=0.0)              # tiles/min
    model_version   = Column(String, default="yolov8n_5fold_wbf_v1")

    boards = relationship("Board", back_populates="session", cascade="all, delete-orphan")


class Board(Base):
    """보드 1장 (10×10 타일 그리드)."""

    __tablename__ = "boards"

    board_id    = Column(String, primary_key=True, default=_uuid)
    session_id  = Column(String, ForeignKey("inspection_sessions.session_id"), nullable=False)
    filename    = Column(String, nullable=False)
    file_path   = Column(String, nullable=True)
    grid_rows   = Column(Integer, default=10)
    grid_cols   = Column(Integer, default=10)

    session = relationship("InspectionSession", back_populates="boards")
    tiles   = relationship("TileInspection", back_populates="board", cascade="all, delete-orphan")


class TileInspection(Base):
    """보드 내 타일 1개 검사 결과."""

    __tablename__ = "tile_inspections"

    tile_id         = Column(String, primary_key=True, default=_uuid)
    board_id        = Column(String, ForeignKey("boards.board_id"), nullable=False)
    row             = Column(Integer, nullable=False)
    col             = Column(Integer, nullable=False)
    verdict         = Column(String, nullable=False)          # PASS|FAIL|REVIEW
    max_confidence  = Column(Float, default=0.0)
    inference_ms    = Column(Float, default=0.0)
    scan_order      = Column(Integer, default=0)
    inspected_at    = Column(DateTime, default=datetime.utcnow)

    board      = relationship("Board", back_populates="tiles")
    detections = relationship("Detection", back_populates="tile", cascade="all, delete-orphan")
    review     = relationship("Review", back_populates="tile", uselist=False, cascade="all, delete-orphan")


class Detection(Base):
    """타일 1개 내 결함 탐지 1건."""

    __tablename__ = "detections"

    detection_id = Column(String, primary_key=True, default=_uuid)
    tile_id      = Column(String, ForeignKey("tile_inspections.tile_id"), nullable=False)
    class_name   = Column(String, nullable=False)
    class_id     = Column(Integer, nullable=False)            # 0~5
    confidence   = Column(Float, nullable=False)
    x1           = Column(Float)
    y1           = Column(Float)
    x2           = Column(Float)
    y2           = Column(Float)

    tile = relationship("TileInspection", back_populates="detections")


class Review(Base):
    """사람 검토 결과 — 타일당 1건 (unique tile_id)."""

    __tablename__ = "reviews"

    review_id           = Column(String, primary_key=True, default=_uuid)
    tile_id             = Column(String, ForeignKey("tile_inspections.tile_id"), nullable=False, unique=True)
    reviewer            = Column(String, nullable=True)
    reviewed_at         = Column(DateTime, default=datetime.utcnow)
    final_verdict       = Column(String, nullable=False)      # PASS|FAIL
    final_defect_class  = Column(String, nullable=True)       # open|short|... (nullable)

    tile = relationship("TileInspection", back_populates="review")


class Model(Base):
    """AI 모델 메타 정보 — 참조용 독립 테이블."""

    __tablename__ = "models"

    model_id        = Column(String, primary_key=True, default=_uuid)
    architecture    = Column(String, nullable=True)           # 'YOLOv8n'
    weight_file     = Column(String, nullable=True)           # 'best_fold_1.pt'
    fold_number     = Column(Integer, nullable=True)          # 1~5
    ensemble_method = Column(String, nullable=True)           # 'WBF'
    recall          = Column(Float, nullable=True)
    map_score       = Column(Float, nullable=True)
