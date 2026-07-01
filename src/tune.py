"""YOLOv8 하이퍼파라미터 튜닝 스크립트.

실행:
    python src/tune.py [--config config.yaml]

흐름:
    1. config 로드 → 경로 확인
    2. data.yaml 에 processed 경로 주입
    3. YOLO 모델 로드 및 유전 알고리즘 기반 파라미터 튜닝
    4. 최적 파라미터(best_hyperparameters.yaml)가 runs/tune 내부에 자동 생성됨
"""

import sys
import shutil
import argparse
import tempfile
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_paths


def build_data_yaml(processed: Path, base_yaml: str = "data.yaml") -> Path:
    """data.yaml 의 path 플레이스홀더를 실제 processed 경로로 채워 임시 파일 반환."""
    with open(base_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(processed.resolve())
    tmp = Path(tempfile.mktemp(suffix=".yaml"))
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return tmp


def main(config_path: str = "config.yaml") -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    paths = get_paths(cfg)
    
    # tune 파라미터 로드
    tc = cfg.get("tune", {})
    if not tc:
        print("[오류] config.yaml 에 'tune' 설정이 없습니다.")
        return

    data_yaml = build_data_yaml(paths["processed"])

    # 모델 초기화
    model = YOLO(tc.get("model", "yolov8n.pt"))

    print(f"\n[tune] 하이퍼파라미터 튜닝을 시작합니다. (반복: {tc.get('iterations', 10)}회)")
    print("[tune] 튜닝은 일반 학습보다 매우 오랜 시간이 소요됩니다.\n")

    # 튜닝 실행
    results = model.tune(
        data=str(data_yaml),
        epochs=tc.get("epochs", 30),
        iterations=tc.get("iterations", 30),
        imgsz=tc.get("imgsz", 640),
        workers=tc.get("workers", 4),
        project=str(paths["runs"]),
        name="tune",
        exist_ok=True,
        use_ray=False,  # Ultralytics 기본 내장 GA 튜너 사용
    )

    print(f"[tune] 튜닝이 완료되었습니다. 결과물은 {paths['runs']}/tune 디렉토리에 저장되었습니다.")

    data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
