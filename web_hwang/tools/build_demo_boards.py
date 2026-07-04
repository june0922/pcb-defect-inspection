"""데모용 가상 보드 조립 도구.

DeepPCB group 의 640×640 크롭들을 N×M 격자로 이어붙여 가상 보드 3종을 생성한다.

사용 (레포 루트에서):
    python web_hwang/tools/build_demo_boards.py [--raw-data <경로>] [--group group00041] [--rows 4] [--cols 4]

생성 결과 (web_hwang/samples/boards/):
    ok_board.jpg     / ok_board_map.json     — _temp(결함없음) 크롭 전체  → OK 시나리오
    ng_board.jpg     / ng_board_map.json     — _test(결함있음) 크롭 전체  → NG 시나리오
    review_board.jpg / review_board_map.json — _temp 앞절반 + _test 뒷절반 → REVIEW 포함 시나리오

※ 가상 보드는 데모용 합성 이미지입니다.
   DeepPCB 원본에는 2D 위치 정보가 없으므로, 같은 그룹 크롭들을 임의로 이어붙인 합성물입니다.
   REVIEW 시나리오는 모델이 충분히 학습되어 애매한 신뢰도를 출력할 때 재현됩니다.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CELL = 640  # DeepPCB 크롭 크기. 업스케일/다운스케일 없이 그대로 사용.


def collect_crops(crop_dir: Path, suffix: str, n: int) -> list[Path]:
    """crop_dir 에서 suffix(_test / _temp)를 가진 크롭 n장을 파일명 순으로 수집."""
    crops = sorted(crop_dir.glob(f"*{suffix}.jpg"))[:n]
    if len(crops) < n:
        raise ValueError(
            f"{crop_dir} 에서 '{suffix}' 크롭 {n}장 필요, {len(crops)}장만 있음"
        )
    return crops


def assemble_board(crop_paths: list[Path], rows: int, cols: int) -> tuple[np.ndarray, list[dict]]:
    """크롭 목록을 rows×cols 격자로 이어붙여 (board_img, cell_meta_list) 반환."""
    assert len(crop_paths) == rows * cols, f"크롭 수({len(crop_paths)}) ≠ {rows}×{cols}"
    board = np.zeros((rows * CELL, cols * CELL, 3), dtype=np.uint8)
    cells: list[dict] = []
    for idx, path in enumerate(crop_paths):
        r, c = divmod(idx, cols)
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"이미지 로드 실패: {path}")
        if img.shape[:2] != (CELL, CELL):
            img = cv2.resize(img, (CELL, CELL))
        board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = img
        cells.append({
            "row": r,
            "col": c,
            "source_file": path.name,
            "cell_type": "temp" if "_temp" in path.name else "test",
        })
    return board, cells


def save_board(
    board_img: np.ndarray,
    cells: list[dict],
    out_dir: Path,
    name: str,
    rows: int,
    cols: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{name}.jpg"
    map_path = out_dir / f"{name}_map.json"
    cv2.imwrite(str(img_path), board_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {
        "board_name": name,
        "grid_rows": rows,
        "grid_cols": cols,
        "cell_size": CELL,
        "cells": cells,
    }
    map_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  ✓ {name}: {rows}×{cols} 보드 → {img_path.name} + {map_path.name}")


def main(raw_data: Path, group: str, rows: int, cols: int, out_dir: Path) -> None:
    n = rows * cols
    group_top = raw_data / group
    if not group_top.exists():
        print(f"[ERROR] 그룹 폴더 없음: {group_top}")
        sys.exit(1)

    # 각 group 하위에 실제 크롭 폴더가 있음 (예: group00041/00041/)
    crop_dirs = [d for d in sorted(group_top.iterdir()) if d.is_dir() and not d.name.endswith("_not")]
    if not crop_dirs:
        print(f"[ERROR] 크롭 폴더 없음: {group_top}")
        sys.exit(1)
    crop_dir = crop_dirs[0]
    print(f"[build] 크롭 폴더: {crop_dir}")
    print(f"[build] 격자: {rows}×{cols} = {n}칸")

    test_crops = collect_crops(crop_dir, "_test", n)
    temp_crops = collect_crops(crop_dir, "_temp", n)

    print("[build] 보드 생성 중...")

    # OK 보드: _temp 크롭 전체 (결함 없는 템플릿)
    board, cells = assemble_board(temp_crops, rows, cols)
    save_board(board, cells, out_dir, "ok_board", rows, cols)

    # NG 보드: _test 크롭 전체 (결함 있는 테스트)
    board, cells = assemble_board(test_crops, rows, cols)
    save_board(board, cells, out_dir, "ng_board", rows, cols)

    # REVIEW 보드: _temp 앞 절반 + _test 뒷 절반 (혼합)
    half = n // 2
    mixed = temp_crops[:half] + test_crops[half:]
    board, cells = assemble_board(mixed, rows, cols)
    save_board(board, cells, out_dir, "review_board", rows, cols)

    print(f"\n[build] 완료 → {out_dir}")
    print("  ok_board     — _temp 크롭 전체   (정상 보드, OK 예상)")
    print("  ng_board     — _test 크롭 전체   (결함 보드, NG 예상)")
    print("  review_board — _temp+_test 혼합  (혼합 보드, REVIEW 포함 예상)")
    print("\n[참고] REVIEW 시나리오는 모델이 충분히 학습되어")
    print("       _test 크롭에서 review_band 수준의 신뢰도를 출력할 때 재현됩니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데모용 가상 보드 조립")
    parser.add_argument(
        "--raw-data",
        default="/Users/hwangminjun/DeepPCB/PCBData",
        help="DeepPCB PCBData 경로 (config.yaml local.raw_data 와 동일)",
    )
    parser.add_argument("--group", default="group00041", help="사용할 그룹명")
    parser.add_argument("--rows", type=int, default=4, help="격자 행 수")
    parser.add_argument("--cols", type=int, default=4, help="격자 열 수")
    parser.add_argument(
        "--out-dir",
        default="web_hwang/samples/boards",
        help="보드 저장 경로 (레포 루트 기준)",
    )
    args = parser.parse_args()

    main(
        raw_data=Path(args.raw_data),
        group=args.group,
        rows=args.rows,
        cols=args.cols,
        out_dir=Path(args.out_dir),
    )
