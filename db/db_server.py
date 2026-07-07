"""PCB 검사 공유 DB API 서버.

실행:
    python -m db.db_server          # 기본 0.0.0.0:8001
    python -m db.db_server --port 8001 --host 0.0.0.0

app_front → POST /session, /board, /tile, PUT /session/{id}/summary
app_back  → GET /review/tiles, POST /review
"""

import argparse
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .init_db import init_db
from . import writer as _writer

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────

app = FastAPI(title="PCB DB Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    print("[DB Server] 시작 완료 — DB 초기화됨")


# ── 헬스 체크 ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── 요청/응답 스키마 ──────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    session_id: Optional[str] = None   # 클라이언트 제공 시 그대로 사용 (폴백 동기화용)
    source: str
    target_folder: str
    pass_threshold: float
    fail_threshold: float
    iou_threshold: float
    model_version: str = "yolov8n_5fold_wbf_v1"


class SessionSummary(BaseModel):
    total_tiles: int
    pass_count: int
    fail_count: int
    review_count: int
    fpy: float
    avg_inference_ms: float
    throughput: float


class BoardCreate(BaseModel):
    board_id: Optional[str] = None     # 클라이언트 제공 시 그대로 사용
    session_id: str
    filename: str
    file_path: str
    grid_rows: int = 10
    grid_cols: int = 10


class DetectionItem(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox_abs: list[float]   # [x1, y1, x2, y2]


class TileCreate(BaseModel):
    tile_id: Optional[str] = None      # 클라이언트 제공 시 그대로 사용
    board_id: str
    row: int
    col: int
    verdict: str
    max_confidence: float
    inference_ms: float
    scan_order: int
    detections: list[DetectionItem] = []


class ReviewUpsert(BaseModel):
    tile_id: str
    final_verdict: str
    final_defect_class: Optional[str] = None
    reviewer: str = "operator"


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.post("/session", status_code=201)
def create_session(body: SessionCreate):
    sid = _writer.create_session(
        source=body.source,
        target_folder=body.target_folder,
        pass_threshold=body.pass_threshold,
        fail_threshold=body.fail_threshold,
        iou_threshold=body.iou_threshold,
        model_version=body.model_version,
        session_id=body.session_id,
    )
    return {"session_id": sid}


@app.put("/session/{session_id}/summary")
def update_session(session_id: str, body: SessionSummary):
    _writer.update_session_summary(
        session_id=session_id,
        total_tiles=body.total_tiles,
        pass_count=body.pass_count,
        fail_count=body.fail_count,
        review_count=body.review_count,
        fpy=body.fpy,
        avg_inference_ms=body.avg_inference_ms,
        throughput=body.throughput,
    )
    return {"ok": True}


@app.post("/board", status_code=201)
def create_board(body: BoardCreate):
    bid = _writer.create_board(
        session_id=body.session_id,
        filename=body.filename,
        file_path=body.file_path,
        grid_rows=body.grid_rows,
        grid_cols=body.grid_cols,
        board_id=body.board_id,
    )
    return {"board_id": bid}


@app.post("/tile", status_code=201)
def create_tile(body: TileCreate):
    dets = [d.model_dump() for d in body.detections]
    tid = _writer.create_tile(
        board_id=body.board_id,
        row=body.row,
        col=body.col,
        verdict=body.verdict,
        max_confidence=body.max_confidence,
        inference_ms=body.inference_ms,
        scan_order=body.scan_order,
        detections=dets,
        tile_id=body.tile_id,
    )
    return {"tile_id": tid}


@app.get("/review/tiles")
def get_review_tiles(
    verdict: str = "FAIL,REVIEW",
    since: Optional[str] = None,
    limit: int = 200,
):
    """리뷰 대기 타일 목록 조회.

    since: ISO 8601 datetime 문자열 — 이 시각 이후 검사된 타일만 반환
    verdict: 쉼표 구분 (예: "FAIL,REVIEW")
    """
    verdict_list = [v.strip() for v in verdict.split(",")]
    tiles = _writer.fetch_tiles_for_review(verdict_list)

    # since 필터
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            since_dt = since_dt.replace(tzinfo=None)
            tiles = [t for t in tiles if _parse_dt(t.get("inspected_at")) > since_dt]
        except Exception:
            pass

    return {"tiles": tiles[:limit]}


@app.post("/review")
def upsert_review(body: ReviewUpsert):
    _writer.upsert_review(
        tile_id=body.tile_id,
        final_verdict=body.final_verdict,
        final_defect_class=body.final_defect_class,
        reviewer=body.reviewer,
    )
    return {"ok": True}


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _parse_dt(s) -> datetime:
    if isinstance(s, datetime):
        return s
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return datetime.min


# ── 직접 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run("db.db_server:app", host=args.host, port=args.port, reload=False)
