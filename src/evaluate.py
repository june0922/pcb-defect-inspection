"""test 세트 평가 스크립트.

실행:
    python src/evaluate.py [--config config.yaml]

출력:
    mAP@0.5 / mAP@0.5:0.95 / recall (recall 강조)
"""

import sys
import argparse
import tempfile
from pathlib import Path

import yaml

# 프로젝트 루트: src/ 의 부모 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.append( str(Path(__file__).parent))
from utils import load_config, get_paths

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


def build_data_yaml(processed: Path, base_yaml: Path | None = None) -> Path:
    """data.yaml path 플레이스홀더를 실제 processed 경로로 채워 임시 파일 반환.

    Args:
        processed: 전처리된 데이터의 절대 경로.
        base_yaml: 기반 data.yaml 경로. None 이면 PROJECT_ROOT/data.yaml 사용.

    Returns:
        생성된 임시 YAML 파일의 Path.
    """
    if base_yaml is None:
        base_yaml = PROJECT_ROOT / "data.yaml"

    with open(base_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(processed.resolve())

    # mktemp() 대신 NamedTemporaryFile 사용 (TOCTOU race condition 방지)
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    ) as tmp_f:
        yaml.dump(data, tmp_f)
        tmp = Path(tmp_f.name)
    return tmp


def print_metrics(metrics) -> None:
    """주요 메트릭 출력. recall 을 가장 먼저 강조."""
    # ultralytics DetMetrics 구조 참고
    r = metrics.box.mr if hasattr(metrics.box, "mr") else float("nan")
    map50 = metrics.box.map50 if hasattr(metrics.box, "map50") else float("nan")
    map5095 = metrics.box.map if hasattr(metrics.box, "map") else float("nan")

    print("\n" + "=" * 50)
    print(f"  Recall (mean)     : {r:.4f}  ← recall 우선 지표")
    print(f"    mAP@0.5           : {map50:.4f}")
    print(f"    mAP@0.5:0.95      : {map5095:.4f}")
    print("=" * 50)

    # TODO: 클래스별 recall / mAP 출력
    # TODO: confusion matrix 저장
    # TODO: FP/FN 이미지 샘플 저장 (recall 분석용)


def main(config_path: str = "config.yaml") -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    paths = get_paths(cfg)

    weight = paths["weights"] / "best.pt"
    if not weight.exists():
        print(f"[ERROR] best.pt 가 없습니다: {weight}")
        sys.exit(1)

    model = YOLO(str(weight))
    data_yaml = build_data_yaml(paths["processed"])

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(paths["runs"]),
        name="eval",
        exist_ok=True,
    )

    print_metrics(metrics)
    data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
