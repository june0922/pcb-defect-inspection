"""PCB 결함 검사 웹 데모 서버 (FastAPI).

실행 (레포 루트에서):
    uvicorn web.app:app --reload --port 8000

또는 web/ 폴더 안에서:
    uvicorn app:app --reload --port 8000

API:
    GET  /              → static/index.html
    GET  /samples       → samples/ 이미지 파일 목록 JSON
    GET  /judge-config  → config.yaml 의 판정 기준 JSON
    POST /inspect       → 추론 + 판정 + 주석 이미지(base64) 반환
"""

import sys
import json
import base64
import tempfile
import importlib.util
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
REPO_ROOT = WEB_DIR.parent
SRC_DIR = REPO_ROOT / "src"
CONFIG_PATH = REPO_ROOT / "config.yaml"
SAMPLES_DIR = WEB_DIR / "samples"
BOARDS_DIR  = SAMPLES_DIR / "boards"
STATIC_DIR  = WEB_DIR / "static"

# TODO(실제연결): 학습 완료 후 'yolov8n.pt' 를 지우고 best.pt 경로를 최우선으로 사용
BEST_MODEL_PATHS = [
    WEB_DIR / "best.pt",
    REPO_ROOT / "weights" / "best.pt",
]
FALLBACK_MODEL = "yolov8n.pt"
 
# ---------------------------------------------------------------------------
# src 모듈 로드
# ---------------------------------------------------------------------------
def _load_src(name: str):
    path = SRC_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pcb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_utils_mod = _load_src("utils")
_inspect_mod = _load_src("pcb_inspect")
_visualize_mod = _load_src("visualize")

load_config            = _utils_mod.load_config
inspect_image          = _inspect_mod.inspect_image
draw_inspection_result = _visualize_mod.draw_inspection_result
extract_defects        = _inspect_mod.extract_defects
judge                  = _inspect_mod.judge

# ---------------------------------------------------------------------------
# 전역 상태 — startup 시 딱 한 번 로드, 요청마다 재로드 금지
# ---------------------------------------------------------------------------
_model = None
_cfg: Optional[dict] = None


def _fmt(d: dict) -> dict:
    """결함 dict 의 center tuple → list 변환 (JSON 직렬화용)."""
    return {**d, "center": list(d["center"])}


def _find_model():
    for p in BEST_MODEL_PATHS:
        if p.exists():
            return str(p)
    # best.pt 없으면 yolov8n.pt 로 폴백 (ultralytics 가 자동 다운로드)
    print(f"[startup] best.pt 없음 → {FALLBACK_MODEL} 사용 (추론 전용)")
    return FALLBACK_MODEL


def _startup() -> None:
    global _model, _cfg
    _cfg = load_config(str(CONFIG_PATH))
    model_path = _find_model()

    from ultralytics import YOLO
    print(f"[startup] 모델 로드 중: {model_path}")
    _model = YOLO(str(model_path))

    # 워밍업: 첫 클릭이 느리면 시연 인상이 나쁘므로, 시작 시 더미 추론 1회 실행
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _model(dummy, verbose=False)
    print("[startup] 워밍업 완료. http://localhost:8000 접속하세요.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup()
    yield  # 서버 실행 구간


app = FastAPI(title="PCB 검사 데모 — Team Convex", lifespan=lifespan)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# sample-files: 썸네일 이미지 src 로 사용 (예: /sample-files/ok_board.jpg)
app.mount("/sample-files", StaticFiles(directory=str(SAMPLES_DIR)), name="sample-files")


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """메인 검사 화면."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/samples")
def list_samples():
    """samples/ 의 이미지 파일 목록을 반환.

    프론트엔드가 이 목록으로 썸네일을 렌더링한다.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(
        f.name
        for f in SAMPLES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    return {"samples": files}


@app.get("/judge-config")
def judge_config():
    """판정 기준값(conf_threshold, review_band)을 반환.

    프론트엔드 오른쪽 패널의 '판정 기준' 섹션에 표시된다.
    recall 우선 설계를 시연 화면에서 바로 확인할 수 있게 한다.
    """
    jcfg = _cfg.get("judge", {})
    return {
        "conf_threshold": jcfg.get("conf_threshold", 0.5),
        "review_band": jcfg.get("review_band", [0.3, 0.5]),
    }


@app.post("/inspect")
def inspect_endpoint(
    source: str = Form(...),
    filename: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """이미지를 받아 YOLO 추론 → inspect.py 판정 → 주석 이미지(base64) 반환.

    판정 로직은 src/inspect.py 에 있으며 여기서 재사용만 한다.

    Args (form):
        source   : "sample" | "upload"
        filename : source="sample" 일 때 samples/ 의 파일명
        file     : source="upload" 일 때 업로드 파일
    """
    tmp_path: Optional[Path] = None

    try:
        # 이미지 경로 결정
        if source == "sample":
            if not filename:
                raise HTTPException(status_code=400, detail="filename 필드가 필요합니다.")
            safe_name = Path(filename).name  # 경로 탐색 공격 방지
            image_path = SAMPLES_DIR / safe_name
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"샘플 파일 없음: {safe_name}")

        elif source == "upload":
            if not file:
                raise HTTPException(status_code=400, detail="file 필드가 필요합니다.")
            suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(file.file.read())
            tmp.close()
            tmp_path = Path(tmp.name)
            image_path = tmp_path

        else:
            raise HTTPException(status_code=400, detail=f"알 수 없는 source: {source!r}")

        # 판정 — 로직은 src/inspect.py 에 위임
        result = inspect_image(str(image_path), _model, _cfg)

        # 주석 이미지 — src/visualize.py 재사용
        annotated: np.ndarray = draw_inspection_result(str(image_path), result)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return JSONResponse({
            "verdict": result["verdict"],
            "defect_count": result["defect_count"],
            "by_class": result["by_class"],
            "defects": [_fmt(d) for d in result["defects"]],
            "review": [_fmt(d) for d in result["review"]],
            "image_b64": img_b64,
        })

    finally:
        # 업로드 임시 파일 정리
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# 보드 검사 라우트
# ---------------------------------------------------------------------------

@app.get("/board")
def board_index():
    """전체 보드 격자 순차 검사 화면."""
    return FileResponse(str(STATIC_DIR / "board.html"))


@app.get("/boards")
def list_boards():
    """생성된 가상 보드 목록 반환 (build_demo_boards.py 로 생성)."""
    if not BOARDS_DIR.exists():
        return {"boards": []}
    boards = [f.stem.replace("_map", "") for f in sorted(BOARDS_DIR.glob("*_map.json"))]
    return {"boards": boards}


@app.post("/inspect_board")
def inspect_board(board_id: str = Form(...)):
    """보드 이미지를 격자로 분할 → 칸별 검사 → 보드 최종 판정.

    판정 집계:
        NG  셀이 하나라도 있으면 → 보드 NG
        NG 없고 REVIEW 셀이 있으면 → 보드 REVIEW
        전체 OK → 보드 OK
    """
    safe_id = Path(board_id).name  # 경로 탐색 방지
    board_path = BOARDS_DIR / f"{safe_id}.jpg"
    map_path   = BOARDS_DIR / f"{safe_id}_map.json"

    if not board_path.exists() or not map_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"보드 없음: {safe_id}. 먼저 python web/tools/build_demo_boards.py 를 실행하세요.",
        )

    meta      = json.loads(map_path.read_text())
    rows      = meta["grid_rows"]
    cols      = meta["grid_cols"]
    cell_size = meta["cell_size"]

    board_img = cv2.imread(str(board_path))
    if board_img is None:
        raise HTTPException(status_code=500, detail="보드 이미지 로드 실패")

    cell_results: list[dict] = []
    for r in range(rows):
        for c in range(cols):
            tile = board_img[r * cell_size:(r + 1) * cell_size,
                             c * cell_size:(c + 1) * cell_size]
            yolo_results = _model(tile, verbose=False)
            defects = extract_defects(yolo_results)
            verdict, review_items = judge(defects, _cfg)
            by_class: dict = {}
            for d in defects:
                by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1
            cell_results.append({
                "row": r,
                "col": c,
                "verdict": verdict,
                "defect_count": len(defects),
                "by_class": by_class,
                "defects": [_fmt(d) for d in defects],
                "review": [_fmt(d) for d in review_items],
            })

    ng_count  = sum(1 for cell in cell_results if cell["verdict"] == "NG")
    rv_count  = sum(1 for cell in cell_results if cell["verdict"] == "REVIEW")
    ok_count  = len(cell_results) - ng_count - rv_count

    if ng_count > 0:
        board_verdict = "NG"
    elif rv_count > 0:
        board_verdict = "REVIEW"
    else:
        board_verdict = "OK"

    return JSONResponse({
        "board_verdict": board_verdict,
        "board_id": safe_id,
        "grid_rows": rows,
        "grid_cols": cols,
        "cell_size": cell_size,
        "cells": cell_results,
        "summary": {"ok": ok_count, "ng": ng_count, "review": rv_count},
    })


@app.get("/board_cell/{board_id}/{row}/{col}")
def board_cell(board_id: str, row: int, col: int):
    """보드에서 특정 칸(row, col)의 이미지를 JPEG 로 반환.

    REVIEW 패널에서 개별 칸 이미지를 보여줄 때 사용한다.
    """
    safe_id   = Path(board_id).name
    board_path = BOARDS_DIR / f"{safe_id}.jpg"
    map_path   = BOARDS_DIR / f"{safe_id}_map.json"

    if not board_path.exists():
        raise HTTPException(status_code=404, detail=f"보드 없음: {safe_id}")

    meta      = json.loads(map_path.read_text())
    cell_size = meta["cell_size"]
    max_r     = meta["grid_rows"] - 1
    max_c     = meta["grid_cols"] - 1

    if not (0 <= row <= max_r and 0 <= col <= max_c):
        raise HTTPException(status_code=400, detail=f"범위 초과: ({row},{col})")

    board_img = cv2.imread(str(board_path))
    tile = board_img[row * cell_size:(row + 1) * cell_size,
                     col * cell_size:(col + 1) * cell_size]
    _, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=bytes(buf), media_type="image/jpeg")
