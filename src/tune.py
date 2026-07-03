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
import os
os.environ["TQDM_FORCE_TTY"] = "1"
import shutil
import argparse
import tempfile
from pathlib import Path

import yaml

# 프로젝트 루트: src/ 의 부모 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_paths


def build_data_yaml(processed: Path, base_yaml: Path | None = None) -> Path:
    """data.yaml 의 path 플레이스홀더를 실제 processed 경로로 채워 임시 파일 반환.

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

    # 모델 초기화 (PROJECT_ROOT 기준 절대 경로로 변환)
    model_path = PROJECT_ROOT / tc.get("model", "weights/yolov8n.pt")
    model = YOLO(str(model_path))

    # 이어하기(resume) 여부 자동 감지
    # Ultralytics Tuner 내부에서 exist_ok 값은 resume 값으로 덮여쓰임.
    # tune_results.ndjson 존재 여부로 이어하기 여부를 결정해 resume=True 를 명시적으로 전달해야 함.
    tune_results_file = paths["runs"] / "tune" / "tune_results.ndjson"
    should_resume = tune_results_file.exists()

    if should_resume:
        completed = sum(1 for line in tune_results_file.open(encoding="utf-8") if line.strip())
        total = tc.get("iterations", 100)
        print(f"\n[tune] 이전 튜닝 결과 발견 — iteration {completed}/{total}에서 이어서 시작합니다.")
    else:
        print(f"\n[tune] 하이퍼파라미터 튜닝을 시작합니다. (반복: {tc.get('iterations', 100)}회, 에포크/회: {tc.get('epochs', 15)})")
    print("[tune] 튜닝은 일반 학습보다 매우 오랜 시간이 소요됩니다.\n")

    from utils import TotalETACallback
    eta_callback = TotalETACallback()
    model.add_callback("on_train_epoch_start", eta_callback.on_train_epoch_start)
    model.add_callback("on_train_batch_end", eta_callback.on_train_batch_end)

    # 튜닝 실행
    # resume=True → Tuner 내부에서 exist_ok=True 로 전환되어 같은 runs/tune/ 디렉토리를 재사용하고
    #               tune_results.ndjson 의 완료 행 수를 세어 다음 iteration 부터 자동으로 재개함.
    results = model.tune(
        data=str(data_yaml),
        epochs=tc.get("epochs", 15),
        iterations=tc.get("iterations", 100),
        imgsz=tc.get("imgsz", 640),
        workers=tc.get("workers", 4),
        project=str(paths["runs"]),
        name="tune",
        resume=should_resume,   # exist_ok=True 대신 resume 을 정확히 사용
        use_ray=False,           # Ultralytics 기본 내장 GA 튜너 사용
    )

    print(f"[tune] 튜닝이 완료되었습니다. 결과물은 {paths['runs']}/tune 디렉토리에 저장되었습니다.")

    data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
