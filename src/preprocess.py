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

[수정 이력]
    - 기존 split_dataset()은 이미지 단위 무작위 stratified split이었으나,
      DeepPCB가 11개 그룹(같은 회로 설계를 촬영한 여러 샘플들)으로 구성되어 있어
      그룹 간 배경/배선 패턴이 유사 → 그룹이 train/val/test에 걸쳐 분산되면
      data leakage 발생. 그룹 단위 greedy 재분할로 수정함.
"""

import sys
import argparse
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_paths

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


def get_group_id(img_path: Path) -> str:
    """이미지 경로에서 그룹 ID 추출.

    예) .../group00041/00041/00041000_test.jpg → "00041"
    파일명 앞 5자리(그룹 폴더명과 동일)를 그룹 ID로 사용.
    """
    stem = img_path.stem.replace("_test", "")
    return stem[:5]


def collect_pairs(raw_data: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    """trainval.txt + test.txt 를 읽어 (image_path, label_path, group_id) 쌍 수집."""
    pairs: list[tuple[Path, Path, str]] = []
    for list_file in ("trainval.txt", "test.txt"):
        txt_path = raw_data / list_file
        if not txt_path.exists():
            print(f"[warn] 목록 파일 없음: {txt_path}")
            continue
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, lbl_rel = line.split()
                stem = Path(img_rel).stem
                img_path = raw_data / Path(img_rel).parent / f"{stem}_test.jpg"
                lbl_path = raw_data / lbl_rel
                if img_path.exists() and lbl_path.exists():
                    group_id = get_group_id(img_path)
                    pairs.append((img_path, lbl_path, group_id))
                else:
                    print(f"[warn] 파일 없음 — img:{img_path.exists()} lbl:{lbl_path.exists()}")
        if limit and len(pairs) >= limit:
            break

    if limit:
        pairs = pairs[:limit]
    print(f"[collect_pairs] 수집된 페어 수: {len(pairs)}")
    return pairs


def convert_label(lbl_path: Path, img_w: int, img_h: int) -> list[str]:
    """DeepPCB 라벨 → YOLO 정규화 xywh 포맷 변환. (기존과 동일, 수정 없음)"""
    yolo_lines: list[str] = []
    with open(lbl_path, encoding="utf-8") as f:
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
    pairs: list[tuple[Path, Path, str]],
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[list, list, list]:
    """그룹 단위 greedy 재분할 (leakage 방지).

    같은 그룹(같은 회로 설계의 여러 샘플)이 train/val/test에 걸쳐
    분산되지 않도록, 그룹을 통째로 하나의 split에만 배정한다.

    Returns:
        (train_pairs, val_pairs, test_pairs)  각 원소는 (img_path, lbl_path, group_id)
    """
    random.seed(random_state)

    # 1) 그룹별로 pair 묶기
    group_to_pairs: dict[str, list] = {}
    for p in pairs:
        group_to_pairs.setdefault(p[2], []).append(p)

    groups = list(group_to_pairs.keys())
    group_sizes = {g: len(v) for g, v in group_to_pairs.items()}
    total = sum(group_sizes.values())

    test_ratio = 1.0 - train_ratio - val_ratio
    targets = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    target_counts = {k: total * v for k, v in targets.items()}

    # 2) 그룹을 크기 내림차순으로 정렬 → 목표치 대비 가장 부족한 split에 우선 배정
    split_pairs = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    groups_sorted = sorted(groups, key=lambda g: -group_sizes[g])
    for g in groups_sorted:
        deficit = {k: target_counts[k] - split_counts[k] for k in targets}
        best_split = max(deficit, key=deficit.get)
        split_pairs[best_split].extend(group_to_pairs[g])
        split_counts[best_split] += group_sizes[g]

    train, val, test = split_pairs["train"], split_pairs["val"], split_pairs["test"]

    # 3) 검증 — 그룹 겹침 없는지 확인
    tg = set(p[2] for p in train)
    vg = set(p[2] for p in val)
    teg = set(p[2] for p in test)
    assert not (tg & vg) and not (tg & teg) and not (vg & teg), "그룹 겹침 발생!"

    print(f"[split] train={len(train)} ({len(train)/total*100:.1f}%), "
          f"val={len(val)} ({len(val)/total*100:.1f}%), "
          f"test={len(test)} ({len(test)/total*100:.1f}%)")
    print(f"[split] ✅ 그룹 겹침 없음 확인 완료")

    return train, val, test


def save_yolo_format(
    split_name: str,
    pairs: list[tuple[Path, Path, str]],
    processed: Path,
    cfg: dict,
) -> None:
    """이미지 + YOLO 라벨을 processed/{images,labels}/{split} 에 저장."""
    img_out = processed / "images" / split_name
    lbl_out = processed / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path, _group_id in pairs:
        if not img_path.exists():
            print(f"[warn] 이미지 없음: {img_path}")
            continue

        shutil.copy2(img_path, img_out / img_path.name)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] 이미지 로드 실패: {img_path}")
            (img_out / img_path.name).unlink(missing_ok=True)
            continue
        h, w = img.shape[:2]

        yolo_lines = convert_label(lbl_path, w, h)
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))

    print(f"[save] {split_name}: {len(pairs)} 샘플 → {img_out}")


def prepare_dataset(project_root: Path, expected_raw_dir: Path) -> Path | None:
    """dataset.zip 파일이 있으면 압축을 풀고 해당 경로를 반환합니다. (기존과 동일)"""
    zip_path = project_root / "dataset.zip"

    if expected_raw_dir.exists():
        print(f"[prepare_dataset] 이미 데이터셋이 압축 해제되어 있습니다: {expected_raw_dir}")
        return expected_raw_dir

    if not zip_path.exists():
        return None

    print(f"[prepare_dataset] {zip_path} 압축 해제 중...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(project_root)

    if expected_raw_dir.exists():
        print(f"[prepare_dataset] 압축 해제 완료: {expected_raw_dir}")
        return expected_raw_dir
    else:
        print(f"[prepare_dataset] 압축 해제 완료했으나 예상 경로({expected_raw_dir})를 찾을 수 없습니다.")
        return project_root / "dataset"


def main(config_path: str = "config.yaml", limit: int | None = None) -> None:
    cfg = load_config(config_path)
    paths = get_paths(cfg)

    raw_data_path = paths["raw_data"]

    project_root = paths.get("project_root", Path("."))
    extracted_path = prepare_dataset(project_root, raw_data_path)
    if extracted_path and extracted_path.exists():
        raw_data_path = extracted_path

    pairs = collect_pairs(raw_data_path, limit=limit)
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
