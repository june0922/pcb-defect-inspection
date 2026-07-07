"""DB API HTTP 클라이언트 — writer.py와 동일한 인터페이스.

연결 우선순위:
    1. HTTP → db_server (정상)
    2. 연결 실패 → 로컬 fallback.db (SQLite) 에 버퍼
    3. 재연결 감지 → fallback.db 미동기화 행을 서버에 재전송 (자동 sync)

사용:
    from db.api_client import client
    sid = client.create_session(...)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from sqlalchemy import Column, String, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .models import Base as _MainBase
from .models import InspectionSession, Board, TileInspection, Detection, Review

_DEFAULT_URL = "http://localhost:8001"
_TIMEOUT = 3.0


# ── 폴백 전용 SQLite — pending_sync 테이블 ────────────────────────────────────

class _FallbackBase(DeclarativeBase):
    pass


class _PendingSync(_FallbackBase):
    """서버 다운 구간에 쌓인 미전송 작업 큐."""
    __tablename__ = "pending_sync"
    sync_id    = Column(String, primary_key=True)
    operation  = Column(String, nullable=False)   # session|board|tile|review|session_summary
    payload    = Column(Text, nullable=False)      # JSON
    created_at = Column(DateTime, nullable=False)
    synced_at  = Column(DateTime, nullable=True)  # None = 미전송


# ── DBClient ─────────────────────────────────────────────────────────────────

class DBClient:
    """HTTP API 우선, 연결 불가 시 로컬 SQLite 폴백 + 재연결 시 자동 동기화."""

    def __init__(self, server_url: str = _DEFAULT_URL):
        self._url = server_url.rstrip("/")
        self._use_local = False
        self._fb_engine = None
        self._fb_session_factory = None

    def set_server_url(self, url: str) -> None:
        self._url = url.rstrip("/")
        # _use_local 은 HTTP 성공/실패 콜백만 변경 — URL 교체로 리셋하지 않음

    # ── 폴백 SQLite 세션 ──────────────────────────────────────────────────────

    def _get_fb_session(self):
        """항상 로컬 db/fallback.db 를 사용하는 세션 반환."""
        if self._fb_engine is None:
            path = Path(__file__).resolve().parent / "fallback.db"
            self._fb_engine = create_engine(
                f"sqlite:///{path}",
                connect_args={"check_same_thread": False},
            )
            _MainBase.metadata.create_all(self._fb_engine)   # 메인 테이블
            _FallbackBase.metadata.create_all(self._fb_engine)  # pending_sync
            self._fb_session_factory = sessionmaker(bind=self._fb_engine)
        return self._fb_session_factory()

    # ── 미전송 큐 기록 ────────────────────────────────────────────────────────

    def _record_pending(self, operation: str, payload: dict) -> None:
        db = self._get_fb_session()
        try:
            db.add(_PendingSync(
                sync_id=str(uuid.uuid4()),
                operation=operation,
                payload=json.dumps(payload, default=str),
                created_at=datetime.utcnow(),
            ))
            db.commit()
        finally:
            db.close()

    # ── 재연결 시 동기화 ──────────────────────────────────────────────────────

    def _try_sync(self) -> None:
        """pending_sync 의 미전송 행을 생성 순서대로 서버에 재전송."""
        db = self._get_fb_session()
        try:
            pending = (
                db.query(_PendingSync)
                .filter_by(synced_at=None)
                .order_by(_PendingSync.created_at)
                .all()
            )
            if not pending:
                return

            print(f"[Sync] 미전송 데이터 {len(pending)}건 동기화 시작")
            synced = 0
            for row in pending:
                try:
                    p = json.loads(row.payload)
                    op = row.operation
                    if op == "session":
                        requests.post(f"{self._url}/session", json=p, timeout=_TIMEOUT).raise_for_status()
                    elif op == "session_summary":
                        sid = p.pop("session_id")
                        requests.put(f"{self._url}/session/{sid}/summary", json=p, timeout=_TIMEOUT).raise_for_status()
                    elif op == "board":
                        requests.post(f"{self._url}/board", json=p, timeout=_TIMEOUT).raise_for_status()
                    elif op == "tile":
                        requests.post(f"{self._url}/tile", json=p, timeout=_TIMEOUT).raise_for_status()
                    elif op == "review":
                        requests.post(f"{self._url}/review", json=p, timeout=_TIMEOUT).raise_for_status()
                    row.synced_at = datetime.utcnow()
                    synced += 1
                except requests.exceptions.ConnectionError:
                    print(f"[Sync] 서버 재단절 — 동기화 중단 ({synced}/{len(pending)}건 완료)")
                    self._use_local = True
                    break
                except Exception as e:
                    print(f"[Sync] 행 전송 실패 ({row.operation}): {e}")
                    break
            db.commit()
            if synced:
                print(f"[Sync] 완료: {synced}/{len(pending)}건")
        finally:
            db.close()

    # ── HTTP 헬퍼 ─────────────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self._url}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        self._on_server_ok()
        return r.json()

    def _put(self, path: str, body: dict) -> dict:
        r = requests.put(f"{self._url}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        self._on_server_ok()
        return r.json()

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(f"{self._url}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _on_server_ok(self) -> None:
        """HTTP 성공 시 호출 — 폴백에서 복구됐으면 sync 실행."""
        if self._use_local:
            print("[API] 서버 재연결 감지 → 로컬 버퍼 동기화")
            self._use_local = False
            self._try_sync()

    def _on_conn_error(self) -> None:
        if not self._use_local:
            print("[API] 서버 연결 실패 → 로컬 SQLite 폴백")
            self._use_local = True

    def is_server_alive(self) -> bool:
        try:
            r = requests.get(f"{self._url}/health", timeout=_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    # ── 폴백 인라인 writer (fallback.db 전용) ────────────────────────────────

    def _fb_write_session(self, sid: str, source: str, target_folder: str,
                          pass_t: float, fail_t: float, iou_t: float, model_v: str) -> None:
        db = self._get_fb_session()
        try:
            if db.get(InspectionSession, sid) is None:
                db.add(InspectionSession(
                    session_id=sid, source=source, started_at=datetime.utcnow(),
                    target_folder=target_folder, pass_threshold=pass_t,
                    fail_threshold=fail_t, iou_threshold=iou_t, model_version=model_v,
                ))
                db.commit()
        finally:
            db.close()

    def _fb_update_session(self, sid: str, total: int, pass_c: int, fail_c: int,
                           review_c: int, fpy: float, inf_ms: float, tput: float) -> None:
        db = self._get_fb_session()
        try:
            row = db.get(InspectionSession, sid)
            if row:
                row.ended_at = datetime.utcnow()
                row.total_tiles = total
                row.pass_count = pass_c
                row.fail_count = fail_c
                row.review_count = review_c
                row.fpy = fpy
                row.avg_inference_ms = inf_ms
                row.throughput = tput
                db.commit()
        finally:
            db.close()

    def _fb_write_board(self, bid: str, sid: str, filename: str, file_path: str,
                        grid_rows: int, grid_cols: int) -> None:
        db = self._get_fb_session()
        try:
            if db.get(Board, bid) is None:
                db.add(Board(board_id=bid, session_id=sid, filename=filename,
                             file_path=file_path, grid_rows=grid_rows, grid_cols=grid_cols))
                db.commit()
        finally:
            db.close()

    def _fb_write_tile(self, tid: str, bid: str, row: int, col: int, verdict: str,
                       max_conf: float, inf_ms: float, scan_order: int, dets: list) -> None:
        db = self._get_fb_session()
        try:
            if db.get(TileInspection, tid) is None:
                db.add(TileInspection(
                    tile_id=tid, board_id=bid, row=row, col=col, verdict=verdict,
                    max_confidence=max_conf, inference_ms=inf_ms, scan_order=scan_order,
                    inspected_at=datetime.utcnow(),
                ))
                for det in dets:
                    bbox = det.get("bbox_abs", [0.0, 0.0, 0.0, 0.0])
                    db.add(Detection(
                        tile_id=tid,
                        class_name=det.get("class_name", ""),
                        class_id=int(det.get("class_id", 0)),
                        confidence=float(det.get("confidence", 0.0)),
                        x1=float(bbox[0]), y1=float(bbox[1]),
                        x2=float(bbox[2]), y2=float(bbox[3]),
                    ))
                db.commit()
        finally:
            db.close()

    def _fb_upsert_review(self, tile_id: str, final_verdict: str,
                          final_defect_class: str | None, reviewer: str) -> None:
        db = self._get_fb_session()
        try:
            row = db.query(Review).filter_by(tile_id=tile_id).first()
            if row:
                row.reviewer = reviewer
                row.reviewed_at = datetime.utcnow()
                row.final_verdict = final_verdict
                row.final_defect_class = final_defect_class
            else:
                db.add(Review(tile_id=tile_id, reviewer=reviewer,
                              reviewed_at=datetime.utcnow(), final_verdict=final_verdict,
                              final_defect_class=final_defect_class))
            db.commit()
        finally:
            db.close()

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def create_session(
        self,
        source: str,
        target_folder: str,
        pass_threshold: float,
        fail_threshold: float,
        iou_threshold: float,
        model_version: str = "yolov8n_5fold_wbf_v1",
    ) -> str:
        sid = str(uuid.uuid4())   # 클라이언트에서 UUID 생성 (폴백/정상 공통)
        payload = dict(session_id=sid, source=source, target_folder=target_folder,
                       pass_threshold=pass_threshold, fail_threshold=fail_threshold,
                       iou_threshold=iou_threshold, model_version=model_version)
        try:
            self._post("/session", payload)
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            self._fb_write_session(sid, source, target_folder,
                                   pass_threshold, fail_threshold, iou_threshold, model_version)
            self._record_pending("session", payload)
        return sid

    def update_session_summary(
        self,
        session_id: str,
        total_tiles: int,
        pass_count: int,
        fail_count: int,
        review_count: int,
        fpy: float,
        avg_inference_ms: float,
        throughput: float,
    ) -> None:
        summary = dict(total_tiles=total_tiles, pass_count=pass_count, fail_count=fail_count,
                       review_count=review_count, fpy=fpy, avg_inference_ms=avg_inference_ms,
                       throughput=throughput)
        try:
            self._put(f"/session/{session_id}/summary", summary)
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            self._fb_update_session(session_id, total_tiles, pass_count, fail_count,
                                    review_count, fpy, avg_inference_ms, throughput)
            self._record_pending("session_summary", {**summary, "session_id": session_id})

    def create_board(
        self,
        session_id: str,
        filename: str,
        file_path: str,
        grid_rows: int,
        grid_cols: int,
    ) -> str:
        bid = str(uuid.uuid4())
        payload = dict(board_id=bid, session_id=session_id, filename=filename,
                       file_path=file_path, grid_rows=grid_rows, grid_cols=grid_cols)
        try:
            self._post("/board", payload)
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            self._fb_write_board(bid, session_id, filename, file_path, grid_rows, grid_cols)
            self._record_pending("board", payload)
        return bid

    def create_tile(
        self,
        board_id: str,
        row: int,
        col: int,
        verdict: str,
        max_confidence: float,
        inference_ms: float,
        scan_order: int,
        detections: list,
    ) -> str:
        tid = str(uuid.uuid4())
        payload = dict(tile_id=tid, board_id=board_id, row=row, col=col, verdict=verdict,
                       max_confidence=max_confidence, inference_ms=inference_ms,
                       scan_order=scan_order, detections=detections)
        try:
            self._post("/tile", payload)
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            self._fb_write_tile(tid, board_id, row, col, verdict, max_confidence,
                                inference_ms, scan_order, detections)
            self._record_pending("tile", payload)
        return tid

    def fetch_new_tiles(
        self,
        verdict: str = "FAIL,REVIEW",
        since: str | None = None,
    ) -> list[dict]:
        params: dict = {"verdict": verdict}
        if since:
            params["since"] = since
        try:
            data = self._get("/review/tiles", params=params)
            return data.get("tiles", [])
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            # 서버 다운 시 PostgreSQL 직접 조회 시도 (같은 네트워크면 가능)
            try:
                from . import writer as _w
                return _w.fetch_tiles_for_review(verdict_filter=verdict.split(","))
            except Exception:
                return []
        except Exception:
            return []

    def upsert_review(
        self,
        tile_id: str,
        final_verdict: str,
        final_defect_class: Optional[str] = None,
        reviewer: str = "operator",
    ) -> None:
        payload = dict(tile_id=tile_id, final_verdict=final_verdict,
                       final_defect_class=final_defect_class, reviewer=reviewer)
        try:
            self._post("/review", payload)
        except requests.exceptions.ConnectionError:
            self._on_conn_error()
            self._fb_upsert_review(tile_id, final_verdict, final_defect_class, reviewer)
            self._record_pending("review", payload)

    def pending_count(self) -> int:
        """미동기화 행 수 반환 — 앱 UI 상태 표시용."""
        try:
            db = self._get_fb_session()
            count = db.query(_PendingSync).filter_by(synced_at=None).count()
            db.close()
            return count
        except Exception:
            return 0


# 싱글턴
client = DBClient()
