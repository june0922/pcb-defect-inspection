"""보드 단위 판정 레이어 차별점

단순 YOLO 검출 결과를 받아 OK / NG / REVIEW 세 가지 판정을 내린다.

판정 로직 (recall 우선):
    - 결함 0개                         → OK
    - conf ≥ conf_threshold 결함 존재   → NG
    - review_band 안 결함만 있음        → REVIEW  (수동 검토 대기)
    - TODO: 클래스별 spec 룰 추가 예정

실행 예:
    python -c "
    from ultralytics import YOLO
    from src.utils import load_config, get_paths
    from web_hwang.pcb_inspect import inspect_image
    cfg = load_config('config.yaml')
    paths = get_paths(cfg)
    model = YOLO(paths['weights'] / 'best.pt')
    print(inspect_image('sample.jpg', model, cfg))
    "
"""

import sys
from pathlib import Path

sys.path.append( str(Path(__file__).parent.parent / "src"))

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]

VERDICT_OK = "OK"
VERDICT_NG = "NG"
VERDICT_REVIEW = "REVIEW"


def extract_defects(results) -> list[dict]:
    """YOLO results 에서 결함 정보를 추출.

    Args:
        results: model(image_path) 의 반환값 (list of Results)

    Returns:
        list of dict, 각 dict:
            class_id  (int)
            class_name (str)
            conf      (float)
            bbox      (list[float])  — xyxy 절대좌표
            center    (tuple[float]) — (cx, cy)
    """
    defects = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            defects.append({
                "class_id": cls_id,
                "class_name": CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id),
                "conf": conf,
                "bbox": xyxy,
                "center": (cx, cy),
            })
    return defects


def judge(defects: list[dict], cfg: dict) -> tuple[str, list[dict]]:
    """결함 목록을 받아 (verdict, review_items) 를 반환.

    Args:
        defects: extract_defects() 반환값
        cfg: config.yaml dict

    Returns:
        verdict    : "OK" | "NG" | "REVIEW"
        review_items: REVIEW 판정 시 애매한 결함 목록 (NG 면 빈 리스트)
    """
    if not defects:
        return VERDICT_OK, []

    jcfg = cfg["judge"]
    conf_thr = jcfg["conf_threshold"]
    band_lo, band_hi = jcfg["review_band"]

    # TODO: 클래스별 spec 룰 (예: pinhole 은 conf_thr 무관 1개도 NG)
    # TODO: 복수 결함 조합 룰 (예: open + short 동시 발생 = NG)

    certain_ng = [d for d in defects if d["conf"] >= conf_thr]
    if certain_ng:
        return VERDICT_NG, []

    review_items = [d for d in defects if band_lo <= d["conf"] < band_hi]
    if review_items:
        return VERDICT_REVIEW, review_items

    # 모든 결함이 band_lo 미만이면 무시 (노이즈 수준으로 간주)
    # recall 우선 철학: band_lo 를 충분히 낮게 설정해야 이 케이스가 드물게 발생
    return VERDICT_OK, []


def inspect_image(image_path: str, model, cfg: dict) -> dict:
    """단일 이미지를 검사하고 현장에서 바로 쓸 수 있는 판정 dict 를 반환.

    Args:
        image_path: 검사할 이미지 경로
        model: 로드된 YOLO 모델 인스턴스
        cfg: load_config() 로 읽은 설정 dict

    Returns:
        {
            "verdict"      : "OK" | "NG" | "REVIEW",
            "defect_count" : int,
            "by_class"     : dict[class_name, count],
            "defects"      : list[dict],  # 전체 결함 상세
            "review"       : list[dict],  # REVIEW 대상 결함 (REVIEW 판정 시만 채워짐)
        }
    """
    results = model(image_path, verbose=False)
    defects = extract_defects(results)
    verdict, review_items = judge(defects, cfg)

    by_class: dict[str, int] = {}
    for d in defects:
        by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1

    return {
        "verdict": verdict,
        "defect_count": len(defects),
        "by_class": by_class,
        "defects": defects,
        "review": review_items,
    }


if __name__ == "__main__":
    import argparse
    from ultralytics import YOLO
    from utils import load_config, get_paths  # noqa

    parser = argparse.ArgumentParser(description="PCB 이미지 단일 판정")
    parser.add_argument("image", help="검사할 이미지 경로")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    weight = paths["weights"] / "best.pt"
    if not weight.exists():
        print(f"[ERROR] best.pt 없음: {weight}")
        sys.exit(1)

    model = YOLO(str(weight))
    result = inspect_image(args.image, model, cfg)

    print("\n" + "=" * 50)
    print(f"  판정     : {result['verdict']}")
    print(f"  결함 수  : {result['defect_count']}")
    print(f"  클래스별 : {result['by_class']}")
    if result["review"]:
        print(f"  REVIEW   : {len(result['review'])}개 (수동 검토 필요)")
    print("=" * 50)
    for d in result["defects"]:
        tag = " [REVIEW]" if d in result["review"] else ""
        print(
            f"    [{d['class_name']:10s}] conf={d['conf']:.3f}"
            f"  bbox={[round(v, 1) for v in d['bbox']]}{tag}"
        )
