"""기존 JSON 로그 파일을 SQLite DB로 마이그레이션.

대상:
    app_front/inspection_log_*.json  → InspectionSession / Board / TileInspection / Detection
    (같은 폴더의) review_results.json → Review (tile_id 매칭 성공한 건만)

실행:
    python -m db.migrate_json_logs [--logs-dir ./app_front] [--dry-run]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from .init_db import init_db
from .database import get_session
from .models import InspectionSession, Board, TileInspection, Detection, Review


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s[:26], fmt[:len(s[:26])])
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None


def _load_json(path: Path) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] JSON 파싱 실패: {path} — {e}")
        return None


# ── 마이그레이션 로직 ─────────────────────────────────────────────────────────

def migrate_inspection_log(log_path: Path, db, dry_run: bool) -> dict[str, str]:
    """inspection_log_*.json 1개를 DB에 삽입.

    Returns:
        {(image_file, row, col) → tile_id} — Review 매칭용 인덱스
    """
    data = _load_json(log_path)
    if not data:
        return {}

    session_info = data.get("inspection_session", {})
    summary = data.get("summary", {})
    tile_results = data.get("tile_results", [])

    if not tile_results:
        print(f"  [skip] 타일 결과 없음: {log_path.name}")
        return {}

    settings = session_info.get("settings", {})
    session_id = session_info.get("session_id") or None

    session_row = InspectionSession(
        session_id=session_id,
        source="app_front",
        started_at=_parse_dt(session_info.get("start_time")),
        ended_at=_parse_dt(session_info.get("end_time")),
        target_folder=session_info.get("source_folder"),
        pass_threshold=settings.get("pass_threshold", 30) / 100.0,
        fail_threshold=settings.get("fail_threshold", 70) / 100.0,
        iou_threshold=settings.get("iou_threshold", 0.45),
        total_tiles=summary.get("total_tiles", 0),
        pass_count=summary.get("pass_count", 0),
        fail_count=summary.get("fail_count", 0),
        review_count=summary.get("review_count", 0),
        fpy=summary.get("first_pass_yield", 0.0),
        avg_inference_ms=summary.get("avg_inference_ms", 0.0),
        throughput=summary.get("throughput_tiles_per_min", 0.0),
        model_version="yolov8n_5fold_wbf_v1",
    )

    # image_file 별로 Board 생성 (중복 제거)
    board_map: dict[str, str] = {}  # filename → board_id
    tile_index: dict[tuple, str] = {}  # (image_file, row, col) → tile_id

    for entry in tile_results:
        img_file = entry.get("image_file", "unknown")
        if img_file not in board_map:
            from uuid import uuid4
            bid = str(uuid4())
            board_map[img_file] = bid

    board_rows = []
    for img_file, board_id in board_map.items():
        # grid 크기는 로그에 없으므로 10×10 기본값
        board_rows.append(Board(
            board_id=board_id,
            session_id=session_row.session_id,
            filename=img_file,
            file_path=None,
            grid_rows=10,
            grid_cols=10,
        ))

    tile_rows = []
    detection_rows = []

    for entry in tile_results:
        from uuid import uuid4
        img_file = entry.get("image_file", "unknown")
        row_n = entry.get("grid_row", 0)
        col_n = entry.get("grid_col", 0)
        tile_id = str(uuid4())

        tile_rows.append(TileInspection(
            tile_id=tile_id,
            board_id=board_map[img_file],
            row=row_n,
            col=col_n,
            verdict=entry.get("verdict", "PASS"),
            max_confidence=entry.get("max_confidence", 0.0),
            inference_ms=entry.get("inference_time_ms", 0.0),
            scan_order=entry.get("tile_id", 0),
            inspected_at=_parse_dt(entry.get("timestamp")),
        ))

        for det in entry.get("detections", []):
            bbox = det.get("bbox_abs", [0, 0, 0, 0])
            detection_rows.append(Detection(
                tile_id=tile_id,
                class_name=det.get("class_name", ""),
                class_id=int(det.get("class_id", 0)),
                confidence=float(det.get("confidence", 0.0)),
                x1=float(bbox[0]), y1=float(bbox[1]),
                x2=float(bbox[2]), y2=float(bbox[3]),
            ))

        tile_index[(img_file, row_n, col_n)] = tile_id

    print(f"  세션 {session_row.session_id[:8]}…  "
          f"보드 {len(board_rows)}개  타일 {len(tile_rows)}개  검출 {len(detection_rows)}건")

    if not dry_run:
        db.add(session_row)
        db.add_all(board_rows)
        db.add_all(tile_rows)
        db.add_all(detection_rows)
        db.commit()

    return tile_index


def migrate_review_json(review_path: Path, tile_index: dict[tuple, str],
                        db, dry_run: bool) -> None:
    """review_results.json을 읽어 tile_id 매칭 후 Review row 삽입."""
    data = _load_json(review_path)
    if not data or not isinstance(data, list):
        return

    DEFECT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
    matched = skipped = 0

    for entry in data:
        img_path = entry.get("image", "")
        img_file = Path(img_path).name
        verdict_raw = entry.get("verdict", "")

        # app_back 저장 형식: verdict = "pass" | "1"~"6"
        if verdict_raw == "pass":
            final_verdict = "PASS"
            final_defect_class = None
        elif verdict_raw.isdigit():
            idx = int(verdict_raw) - 1
            final_verdict = "FAIL"
            final_defect_class = DEFECT_CLASSES[idx] if 0 <= idx < len(DEFECT_CLASSES) else None
        else:
            skipped += 1
            continue

        # tile_id 매칭: app_back은 row/col을 저장하지 않으므로 bbox 기반 근사 매칭 불가.
        # image_file 기준으로 첫 번째 타일에 매핑 (단건 이미지 리뷰 용도)
        matched_tile_id = None
        for (f, r, c), tid in tile_index.items():
            if f == img_file:
                matched_tile_id = tid
                break

        if not matched_tile_id:
            print(f"  [skip] tile 매칭 실패: {img_file}")
            skipped += 1
            continue

        if not dry_run:
            existing = db.query(Review).filter_by(tile_id=matched_tile_id).first()
            if existing:
                existing.final_verdict = final_verdict
                existing.final_defect_class = final_defect_class
                existing.reviewed_at = datetime.utcnow()
            else:
                db.add(Review(
                    tile_id=matched_tile_id,
                    reviewer="operator",
                    reviewed_at=datetime.utcnow(),
                    final_verdict=final_verdict,
                    final_defect_class=final_defect_class,
                ))
        matched += 1

    if not dry_run:
        db.commit()

    print(f"  리뷰 매칭: {matched}건 성공, {skipped}건 스킵")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run(logs_dir: Path, archive_dir: Path, dry_run: bool) -> None:
    if not dry_run:
        init_db()

    log_files = sorted(logs_dir.glob("inspection_log_*.json"))
    review_files = sorted(logs_dir.glob("review_results*.json"))

    if not log_files:
        print(f"[migrate] 마이그레이션할 JSON 로그 없음: {logs_dir}")
        return

    print(f"[migrate] 검사 로그 {len(log_files)}개, 리뷰 파일 {len(review_files)}개 발견")

    db = get_session() if not dry_run else None
    all_tile_index: dict[tuple, str] = {}

    try:
        for log_path in log_files:
            print(f"\n처리 중: {log_path.name}")
            tile_index = migrate_inspection_log(log_path, db, dry_run)
            all_tile_index.update(tile_index)

        for rev_path in review_files:
            print(f"\n리뷰 처리 중: {rev_path.name}")
            migrate_review_json(rev_path, all_tile_index, db, dry_run)

    finally:
        if db:
            db.close()

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in log_files + review_files:
            dest = archive_dir / f.name
            shutil.move(str(f), str(dest))
            print(f"[archive] {f.name} → {dest}")

    print(f"\n[migrate] {'(dry-run) ' if dry_run else ''}완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="./app_front",
                        help="inspection_log_*.json 이 있는 디렉토리")
    parser.add_argument("--archive-dir", default="./logs/archived",
                        help="원본 JSON 이동 대상 디렉토리")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 DB/파일 변경 없이 처리 내용만 출력")
    args = parser.parse_args()

    run(
        logs_dir=Path(args.logs_dir),
        archive_dir=Path(args.archive_dir),
        dry_run=args.dry_run,
    )
