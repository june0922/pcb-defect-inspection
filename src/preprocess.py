"""DeepPCB raw 데이터 → YOLO 포맷 변환 파이프라인.

실행:
    python src/preprocess.py [--config config.yaml] [--limit N]

흐름:
    collect_pairs() → split_dataset() → save_yolo_format()

DeepPCB 포맷 메모:
    - raw_data/trainval.txt, test.txt: "img_rel lbl_rel" 한 줄씩
    - 실제 이미지 파일명: {stem}_test.jpg  (목록의 {stem}.jpg 에 _test 추가)
    - 라벨 포맷: "x1 y1 x2 y2 type" (공백, 절대 픽셀, type 1~6)
    - 클래스 매핑: type - 1  →  0=open 1=short 2=mousebite 3=spur 4=copper 5=pinhole
"""

import sys
import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_paths

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


def collect_pairs(raw_data: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    """trainval.txt + test.txt 를 읽어 (image_path, label_path) 쌍 수집.

    실제 이미지 파일명은 목록의 {stem}.jpg 에 _test 접미사를 붙인 형태.
    예) trainval.txt: group00041/00041/00041000.jpg
        실제 파일:     group00041/00041/00041000_test.jpg

    Args:
        raw_data: PCBData/ 경로 (trainval.txt, test.txt 가 있는 위치)
        limit:    처리할 최대 페어 수 (None = 전체, 빠른 검증용)
    """
    pairs: list[tuple[Path, Path]] = []
    for list_file in ("trainval.txt", "test.txt"):
        txt_path = raw_data / list_file
        if not txt_path.exists():
            print(f"[warn] 목록 파일 없음: {txt_path}")
            continue
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, lbl_rel = line.split()
                stem = Path(img_rel).stem
                img_path = raw_data / Path(img_rel).parent / f"{stem}_test.jpg"
                lbl_path = raw_data / lbl_rel
                if img_path.exists() and lbl_path.exists():
                    pairs.append((img_path, lbl_path))
                else:
                    print(f"[warn] 파일 없음 — img:{img_path.exists()} lbl:{lbl_path.exists()}")
        if limit and len(pairs) >= limit:
            break

    if limit:
        pairs = pairs[:limit]
    print(f"[collect_pairs] 수집된 페어 수: {len(pairs)}")
    return pairs


def convert_label(lbl_path: Path, img_w: int, img_h: int) -> list[str]:
    """DeepPCB 라벨 → YOLO 정규화 xywh 포맷 변환.

    DeepPCB 포맷: "x1 y1 x2 y2 type" (공백구분, 절대 픽셀 좌표, type 1~6)
    YOLO 포맷:    "cls cx cy w h"     (공백구분, 정규화 0~1)

    type 1~6  →  cls 0~5  (단순 -1 매핑)
    """
    yolo_lines: list[str] = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            x1, y1, x2, y2, type_id = (int(p) for p in parts)
            cls_id = type_id - 1  # 1~6 → 0~5
            if not (0 <= cls_id < len(CLASSES)):
                print(f"[warn] 알 수 없는 class type {type_id} in {lbl_path}")
                continue
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def split_dataset(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[list, list, list]:
    """70/20/10 split (train/val/test).

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val, test = train_test_split(pairs, test_size=test_ratio, random_state=random_state)
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=random_state)
    print(f"[split] train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_yolo_format(
    split_name: str,
    pairs: list[tuple[Path, Path]],
    processed: Path,
    cfg: dict,
) -> None:
    """이미지 + YOLO 라벨을 processed/{images,labels}/{split} 에 저장.

    원본 이미지를 그대로 복사하고 라벨만 YOLO 포맷으로 변환한다.
    서버 /shared raw_data 는 읽기만 하고, 결과는 processed 에만 씀.
    """
    img_out = processed / "images" / split_name
    lbl_out = processed / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        if not img_path.exists():
            print(f"[warn] 이미지 없음: {img_path}")
            continue

        shutil.copy2(img_path, img_out / img_path.name)

        # 이미지 크기를 파일명에서 읽지 않고 실제로 확인
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] 이미지 로드 실패: {img_path}")
            (img_out / img_path.name).unlink(missing_ok=True)
            continue
        h, w = img.shape[:2]

        yolo_lines = convert_label(lbl_path, w, h)
        # YOLO는 이미지와 동일한 stem 으로 라벨을 탐색하므로 img_path.stem 사용
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))

    print(f"[save] {split_name}: {len(pairs)} 샘플 → {img_out}")


def main(config_path: str = "config.yaml", limit: int | None = None) -> None:
    cfg = load_config(config_path)
    paths = get_paths(cfg)

    pairs = collect_pairs(paths["raw_data"], limit=limit)
    if not pairs:
        print("[ERROR] 페어를 찾지 못했습니다.")
        sys.exit(1)

    sp = cfg["split"]
    train, val, test = split_dataset(pairs, sp["train"], sp["val"], sp["random_state"])

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        save_yolo_format(name, subset, paths["processed"], cfg)

    print("[preprocess] 완료. 결과:", paths["processed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="처리할 최대 샘플 수 (스모크 테스트용, 예: --limit 50)",
    )
    args = parser.parse_args()
    main(args.config, limit=args.limit)
