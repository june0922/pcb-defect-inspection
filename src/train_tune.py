"""YOLOv8 학습 스크립트 (Hyperparameter tuned).

실행:
    python src/train_tune.py [--config config.yaml]
    (Windows 환경 통합 실행: scripts\run_train_tune.bat)

흐름:
    1. config 로드 → 경로 확인
    2. data.yaml 에 processed 경로 주입
    3. YOLO 모델 로드 및 학습
    4. best.pt → weights/ 복사
"""

import sys
import os
os.environ["TQDM_FORCE_TTY"] = "1"
import shutil
import argparse
import tempfile
from pathlib import Path

# 프로젝트 루트: src/ 의 부모 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

import torch

import yaml

sys.path.append( str(Path(__file__).parent))
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
    # mktemp() 대신 NamedTemporaryFile 사용 (TOCTOU 경쟁 조건 방지)
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    ) as tmp_f:
        yaml.dump(data, tmp_f)
        tmp = Path(tmp_f.name)
    return tmp


def main(config_path: str = "config.yaml", resume: bool = False) -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    paths = get_paths(cfg)
    tc = cfg["train_tune"]

    # --- 디바이스 확인 및 출력 ---
    device = tc.get("device", 0)  # config.yaml의 train_tune.device (기본값: 0=GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] ✅ GPU 사용: {gpu_name}")
    else:
        print("[device] ⚠️  GPU를 찾을 수 없습니다. CPU로 학습합니다.")
        print("         Colab의 경우 '런타임 > 런타임 유형 변경 > T4 GPU'를 선택하세요.")
        device = "cpu"

    data_yaml = build_data_yaml(paths["processed"])

    # 이어학습(Resume)인 경우 last.pt 가중치 불러오기
    if resume:
        last_pt = paths["runs"] / "train_tune" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[Error] 이어학습을 할 수 없습니다. 마지막 체크포인트({last_pt})를 찾지 못했습니다.")
            sys.exit(1)

        # --- GradScaler 상태 패치 ---
        # CPU로 저장된 체크포인트는 GradScaler state_dict가 {} (빈 딕셔너리).
        # GPU에서 이어학습 시 load_state_dict({})를 호출하면 KeyError: 'scale' 발생.
        # 조건: "scale" 키가 없으면 무조건 패치 (빈 딕셔너리 및 잘못된 키 모두 포함)
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        scaler_state = ckpt.get("scaler", {})
        if "scale" not in scaler_state:
            print("[resume] ⚠️  체크포인트의 GradScaler 상태가 올바르지 않습니다 (CPU 저장본 또는 잘못된 패치).")
            print("[resume]    GPU 이어학습을 위해 GradScaler 상태를 기본값으로 패치합니다...")
            ckpt["scaler"] = {
                "scale": 65536.0,       # PyTorch GradScaler 직렬화 키 (언더스코어 없음)
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 0,   # 이 키만 언더스코어 유지
            }
            torch.save(ckpt, str(last_pt))
            print("[resume] ✅ 체크포인트 패치 완료. 이어학습을 시작합니다...")
        else:
            print(f"[resume] ✅ [{last_pt}] 파일로부터 이어학습을 시작합니다...")

        model = YOLO(str(last_pt))
    else:
        model = YOLO(str(PROJECT_ROOT / tc["model"]))

    from utils import GlobalProgressCallback
    eta_callback = GlobalProgressCallback(total_epochs_per_run=tc["epochs"], total_runs=1, run_type="Train Tune")
    model.add_callback("on_pretrain_routine_end", eta_callback.on_pretrain_routine_end)
    model.add_callback("on_train_batch_end", eta_callback.on_train_batch_end)
    model.add_callback("on_val_end", eta_callback.on_val_end)
    model.add_callback("on_train_end", eta_callback.on_train_end)
    # 모델에 전달할 하이퍼파라미터에서 내부 처리용 키워드 제거
    train_args = {k: v for k, v in tc.items() if k not in ["model", "device"]}

    results = model.train(
        data=str(data_yaml),
        device=device,
        project=str(paths["runs"]),
        name="train_tune",
        exist_ok=True,
        resume=resume,
        verbose=False,
        **train_args
    )

    # best.pt 를 weights/ 로 복사
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = paths["weights"] / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"[train_tune] best.pt 저장 완료: {best_dst}")
    else:
        print(f"[warn] best.pt 를 찾지 못했습니다: {best_src}")

    data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    args = parser.parse_args()
    main(args.config, args.resume)
