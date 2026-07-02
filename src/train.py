"""YOLOv8 학습 스크립트.

실행:
    python src/train.py [--config config.yaml]
    (Windows 환경 통합 실행: scripts\run_train.bat)

흐름:
    1. config 로드 → 경로 확인
    2. data.yaml 에 processed 경로 주입
    3. YOLO 모델 로드 및 학습
    4. best.pt → weights/ 복사
"""

import sys
import shutil
import argparse
import tempfile
from pathlib import Path

import torch

import yaml

sys.path.append( str(Path(__file__).parent))
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


def main(config_path: str = "config.yaml", resume: bool = False) -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    paths = get_paths(cfg)
    tc = cfg["train"]

    # --- 디바이스 확인 및 출력 ---
    device = tc.get("device", 0)  # config.yaml의 train.device (기본값: 0=GPU)
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
        last_pt = paths["runs"] / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[Error] 이어학습을 할 수 없습니다. 마지막 체크포인트({last_pt})를 찾지 못했습니다.")
            sys.exit(1)

        # --- GradScaler 상태 패치 ---
        # CPU로 저장된 체크포인트는 GradScaler 상태가 비어 있어,
        # GPU에서 이어학습 시 "source state dict is empty" RuntimeError가 발생함.
        # 이를 방지하기 위해 비어 있는 scaler 상태를 GPU 기본값으로 교체한다.
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        if not ckpt.get("scaler"):
            print("[resume] ⚠️  체크포인트의 GradScaler 상태가 비어 있습니다 (CPU 저장본).")
            print("[resume]    GPU 이어학습을 위해 GradScaler 상태를 기본값으로 패치합니다...")
            ckpt["scaler"] = {
                "scale": 65536.0,       # PyTorch GradScaler 기본 초기값
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
        model = YOLO(tc["model"])

    # TODO(찾기): lr0, lrf, optimizer (SGD/Adam/AdamW) 파라미터 추가
    # TODO(찾기): augmentation (mosaic, flipud, fliplr, hsv_h/s/v) 설정
    results = model.train(
        data=str(data_yaml),
        epochs=tc["epochs"],
        batch=tc["batch"],
        imgsz=tc["imgsz"],
        workers=tc.get("workers", 4),
        patience=tc.get("patience", 50),  # TODO(찾기): 조기 종료 patience
        device=device,                    # GPU(0) 또는 CPU 명시
        project=str(paths["runs"]),
        name="train",
        exist_ok=True,
        resume=resume,  # ultralytics 내부적으로도 resume 활성화
    )

    # best.pt 를 weights/ 로 복사
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = paths["weights"] / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"[train] best.pt 저장 완료: {best_dst}")
    else:
        print(f"[warn] best.pt 를 찾지 못했습니다: {best_src}")

    data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    args = parser.parse_args()
    main(args.config, args.resume)
