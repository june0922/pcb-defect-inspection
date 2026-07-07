"""DB write 헬퍼 — 각 앱에서 import해서 쓰는 단순 함수 모음.

모든 함수는 세션을 자체적으로 열고 닫아서 호출 측에서 세션 관리를 신경 쓰지 않아도 됨.
예외는 caller 쪽으로 전파되므로, 앱에서 try/except로 감싸 DB 장애가 앱을 멈추지 않도록 할 것.
"""

import uuid
from datetime import datetime

from .database import get_session
from .models import InspectionSession, Board, TileInspection, Detection, Review
from .init_db import init_db

_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True


# ── 세션 ─────────────────────────────────────────────────────────────────────

def create_session(
    source: str,
    target_folder: str,
    pass_threshold: float,
    fail_threshold: float,
    iou_threshold: float,
    model_version: str = "yolov8n_5fold_wbf_v1",
    session_id: str | None = None,
) -> str:
    """새 InspectionSession row 삽입 후 session_id 반환.

    session_id 제공 시 해당 ID 사용 (폴백 동기화용 idempotent).
    """
    _ensure_init()
    if session_id is None:
        session_id = str(uuid.uuid4())
    db = get_session()
    try:
        if db.get(InspectionSession, session_id) is None:
            db.add(InspectionSession(
                session_id=session_id,
                source=source,
                started_at=datetime.utcnow(),
                target_folder=target_folder,
                pass_threshold=pass_threshold,
                fail_threshold=fail_threshold,
                iou_threshold=iou_threshold,
                model_version=model_version,
            ))
            db.commit()
    finally:
        db.close()
    return session_id


def update_session_summary(
    session_id: str,
    total_tiles: int,
    pass_count: int,
    fail_count: int,
    review_count: int,
    fpy: float,
    avg_inference_ms: float,
    throughput: float,
) -> None:
    """세션 종료 시 통계 요약 컬럼 업데이트."""
    db = get_session()
    try:
        row = db.get(InspectionSession, session_id)
        if row:
            row.ended_at = datetime.utcnow()
            row.total_tiles = total_tiles
            row.pass_count = pass_count
            row.fail_count = fail_count
            row.review_count = review_count
            row.fpy = fpy
            row.avg_inference_ms = avg_inference_ms
            row.throughput = throughput
            db.commit()
    finally:
        db.close()


# ── 보드 ─────────────────────────────────────────────────────────────────────

def create_board(
    session_id: str,
    filename: str,
    file_path: str,
    grid_rows: int,
    grid_cols: int,
    board_id: str | None = None,
) -> str:
    """새 Board row 삽입 후 board_id 반환 (idempotent)."""
    if board_id is None:
        board_id = str(uuid.uuid4())
    db = get_session()
    try:
        if db.get(Board, board_id) is None:
            db.add(Board(
                board_id=board_id,
                session_id=session_id,
                filename=filename,
                file_path=file_path,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            ))
            db.commit()
    finally:
        db.close()
    return board_id


# ── 타일 + 검출 ───────────────────────────────────────────────────────────────

def create_tile(
    board_id: str,
    row: int,
    col: int,
    verdict: str,
    max_confidence: float,
    inference_ms: float,
    scan_order: int,
    detections: list,
    tile_id: str | None = None,
) -> str:
    """TileInspection + Detection 행을 한 트랜잭션으로 삽입 후 tile_id 반환 (idempotent).

    detections 원소 형식:
        {"class_name": str, "class_id": int, "confidence": float,
         "bbox_abs": [x1, y1, x2, y2]}
    """
    if tile_id is None:
        tile_id = str(uuid.uuid4())
    db = get_session()
    try:
        if db.get(TileInspection, tile_id) is None:
            db.add(TileInspection(
                tile_id=tile_id,
                board_id=board_id,
                row=row,
                col=col,
                verdict=verdict,
                max_confidence=max_confidence,
                inference_ms=inference_ms,
                scan_order=scan_order,
                inspected_at=datetime.utcnow(),
            ))
            for det in detections:
                bbox = det.get("bbox_abs", [0.0, 0.0, 0.0, 0.0])
                db.add(Detection(
                    tile_id=tile_id,
                    class_name=det.get("class_name", ""),
                    class_id=int(det.get("class_id", 0)),
                    confidence=float(det.get("confidence", 0.0)),
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                ))
            db.commit()
    finally:
        db.close()
    return tile_id


# ── 리뷰 ─────────────────────────────────────────────────────────────────────

def upsert_review(
    tile_id: str,
    final_verdict: str,
    final_defect_class: str | None = None,
    reviewer: str = "operator",
) -> None:
    """Review row UPSERT — tile_id가 이미 있으면 update, 없으면 insert."""
    db = get_session()
    try:
        row = db.query(Review).filter_by(tile_id=tile_id).first()
        if row:
            row.reviewer = reviewer
            row.reviewed_at = datetime.utcnow()
            row.final_verdict = final_verdict
            row.final_defect_class = final_defect_class
        else:
            db.add(Review(
                tile_id=tile_id,
                reviewer=reviewer,
                reviewed_at=datetime.utcnow(),
                final_verdict=final_verdict,
                final_defect_class=final_defect_class,
            ))
        db.commit()
    finally:
        db.close()


# ── 조회 (app_back 용) ────────────────────────────────────────────────────────

def fetch_tiles_for_review(verdict_filter: list | None = None) -> list[dict]:
    """FAIL/REVIEW 타일 목록을 최신 세션 기준으로 조회.

    Returns:
        [{"tile_id": ..., "board_filename": ..., "row": ..., "col": ...,
          "verdict": ..., "max_confidence": ..., "detections": [...]}, ...]
    """
    db = get_session()
    try:
        query = db.query(TileInspection)
        if verdict_filter:
            query = query.filter(TileInspection.verdict.in_(verdict_filter))
        tiles = query.order_by(TileInspection.inspected_at.desc()).limit(500).all()

        result = []
        for t in tiles:
            result.append({
                "tile_id": t.tile_id,
                "board_id": t.board_id,
                "board_filename": t.board.filename if t.board else "",
                "board_file_path": t.board.file_path if t.board else "",
                "grid_rows": t.board.grid_rows if t.board else 10,
                "grid_cols": t.board.grid_cols if t.board else 10,
                "row": t.row,
                "col": t.col,
                "verdict": t.verdict,
                "max_confidence": t.max_confidence,
                "inference_ms": t.inference_ms,
                "inspected_at": t.inspected_at.isoformat() if t.inspected_at else None,
                "detections": [
                    {
                        "class_name": d.class_name,
                        "class_id": d.class_id,
                        "confidence": d.confidence,
                        "bbox_abs": [d.x1, d.y1, d.x2, d.y2],
                    }
                    for d in t.detections
                ],
            })
        return result
    finally:
        db.close()
