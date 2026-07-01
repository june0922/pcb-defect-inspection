"""공통 유틸리티 — config 로드 및 환경별 경로 분기."""

import sys
from pathlib import Path
import yaml

sys.path.append( str(Path(__file__).parent))


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml 을 로드해 dict 로 반환."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> dict[str, Path]:
    """env 분기에 따라 절대 경로 dict 를 반환.

    반환 키:
        raw_data    — DeepPCB 원본 (읽기전용, 서버에서는 /shared 참조)
        processed   — YOLO 포맷으로 변환된 데이터 저장 위치
        weights     — best.pt 저장 위치
        runs        — YOLO 학습 로그/결과
    """
    env = cfg["env"]
    env_paths = cfg["paths"][env]

    raw_data = Path(env_paths["raw_data"])
    project_root = Path(env_paths["project_root"])

    paths = {
        "raw_data": raw_data,
        "project_root": project_root,
        "processed": project_root / "data" / "processed",
        "weights": project_root / "weights",
        "runs": project_root / "runs",
    }

    # processed / weights / runs 는 필요 시 자동 생성 (raw_data 는 건드리지 않음)
    for key in ("processed", "weights", "runs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths
