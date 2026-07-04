"""YOLOv8 K-Fold 교차 검증 학습 스크립트.

실행:
    python src/train_kfold.py [--config config.yaml] [--resume]

--resume 플래그:
    완료된 fold(best_fold_N.pt 존재) → 건너뜀
    중단된 fold(last.pt 존재)       → last.pt 에서 이어 학습
    미시작 fold                      → 처음부터 학습
"""

import sys
import os
os.environ["TQDM_FORCE_TTY"] = "1"
import shutil
import argparse
import tempfile
from collections import Counter
from pathlib import Path

# 프로젝트 루트: src/ 의 부모 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

import torch
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_paths


def get_image_and_label_paths(processed_dir: Path):
    """train, val 폴더의 모든 이미지와 라벨 경로를 스캔하여 반환."""
    images = []
    labels = []
    for split in ["train", "val"]:
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        if img_dir.exists():
            for img_path in img_dir.glob("*.jpg"):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    images.append(img_path)
                    labels.append(lbl_path)
    return images, labels


def get_representative_classes(labels: list[Path], num_classes: int = 6):
    """각 이미지의 라벨 파일에서 가장 출현 빈도가 적은 클래스를 대표 클래스로 추출."""
    # 1. 모든 클래스 출현 빈도 조사
    overall_counts = Counter()
    file_classes = []
    for lbl_path in labels:
        classes_in_file = set()
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                classes_in_file.add(cls_id)
                overall_counts[cls_id] += 1
        file_classes.append(list(classes_in_file))
        
    # 2. 이미지별로 가장 희귀한 클래스를 대표로 지정
    y = []
    for classes in file_classes:
        if not classes:
            y.append(0)  # 배경 또는 라벨 없음
        else:
            # 빈도가 가장 낮은 클래스 선택
            rarest = min(classes, key=lambda c: overall_counts.get(c, float('inf')))
            y.append(rarest)
    return y


def build_kfold_yaml(
    train_txt: Path,
    val_txt: Path,
    base_yaml: Path | None = None,
) -> Path:
    """data.yaml 템플릿을 읽어 동적인 Fold 설정 YAML을 생성합니다.

    Args:
        train_txt: 학습 이미지 경로 목록 txt 파일 (절대 경로)
        val_txt:   검증 이미지 경로 목록 txt 파일 (절대 경로)
        base_yaml: 기반 data.yaml 경로. None 이면 PROJECT_ROOT/data.yaml 사용.

    Returns:
        생성된 임시 YAML 파일의 Path.
    """
    if base_yaml is None:
        base_yaml = PROJECT_ROOT / "data.yaml"

    with open(base_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # train / val 을 절대 경로 txt 파일로 직접 지정
    # (YOLO 는 txt 파일 내 경로 목록을 image list 로 인식)
    data["train"] = str(train_txt.resolve())
    data["val"] = str(val_txt.resolve())
    # path 필드는 txt 절대 경로 방식에서는 불필요하므로 제거
    data.pop("path", None)

    # mktemp() 대신 NamedTemporaryFile 사용 (TOCTOU race condition 방지)
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    ) as tmp_f:
        yaml.dump(data, tmp_f)
        tmp = Path(tmp_f.name)
    return tmp


def patch_grad_scaler(last_pt: Path) -> None:
    """CPU 저장 체크포인트의 GradScaler 상태가 없을 경우 기본값으로 패치.

    GPU 이어학습 시 load_state_dict({}) 호출로 KeyError: 'scale' 발생하는 문제를 방지.
    """
    ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
    scaler_state = ckpt.get("scaler", {})
    if "scale" not in scaler_state:
        print("[resume] ⚠️  GradScaler 상태가 올바르지 않습니다 (CPU 저장본 또는 잘못된 패치).")
        print("[resume]    GPU 이어학습을 위해 GradScaler 상태를 기본값으로 패치합니다...")
        ckpt["scaler"] = {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        }
        torch.save(ckpt, str(last_pt))
        print("[resume] ✅ 체크포인트 패치 완료.")
    else:
        print(f"[resume] ✅ GradScaler 상태 정상 확인.")


def main(config_path: str = "config.yaml", resume: bool = False) -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    paths = get_paths(cfg)
    tc = cfg["train"]
    kf_cfg = cfg.get("kfold", {"k": 5, "random_state": 42})

    k = kf_cfg["k"]
    random_state = kf_cfg["random_state"]

    # --- 디바이스 확인 및 출력 ---
    device = tc.get("device", 0)  # config.yaml의 train.device (기본값: 0=GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] ✅ GPU 사용: {gpu_name}")
    else:
        print("[device] ⚠️  GPU를 찾을 수 없습니다. CPU로 학습합니다.")
        print("         Colab의 경우 '런타임 > 런타임 유형 변경 > T4 GPU'를 선택하세요.")
        device = "cpu"
    
    processed_dir = paths["processed"]
    images, labels = get_image_and_label_paths(processed_dir)
    
    if not images:
        print("[error] 학습할 이미지를 찾을 수 없습니다. 전처리를 먼저 수행하세요.")
        return
        
    print(f"[kfold] 전체 데이터 수: {len(images)} (train + val 통합)")
    
    # 층화를 위한 타겟 라벨 추출
    y = get_representative_classes(labels)
    
    # Stratified K-Fold 분할
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    images_arr = np.array(images)
    
    from utils import GlobalProgressCallback
    eta_callback = GlobalProgressCallback(total_epochs_per_run=tc["epochs"], total_runs=k, run_type="Fold")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images_arr, y)):
        fold_num = fold + 1  # 사용자 노출용 1-indexed 폴드 번호

        # --- Resume 모드: fold 상태 판별 ---
        best_dst = paths["weights"] / f"best_fold_{fold_num}.pt"
        last_pt  = paths["runs"] / "kfold" / f"fold_{fold_num}" / "weights" / "last.pt"

        if resume and best_dst.exists():
            print(f"\n[kfold] Fold {fold_num}/{k} → 이미 완료됨 ({best_dst.name} 존재). 건너뜁니다.")
            continue

        fold_resume = resume and last_pt.exists()

        print(f"\n{'='*40}\n[kfold] Fold {fold_num}/{k} 학습 시작\n{'='*40}")
        if fold_resume:
            print(f"[resume] last.pt 발견: {last_pt}")
            print(f"[resume] 이어 학습을 시작합니다...")
        
        train_imgs = images_arr[train_idx]
        val_imgs = images_arr[val_idx]
        
        # fold용 txt 파일 작성
        train_txt = processed_dir / f"train_fold_{fold}.txt"
        val_txt = processed_dir / f"val_fold_{fold}.txt"
        
        with open(train_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(str(p.resolve()) for p in train_imgs))
        with open(val_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(str(p.resolve()) for p in val_imgs))
            
        # fold용 임시 yaml 작성
        data_yaml = build_kfold_yaml(train_txt, val_txt)

        # 모델 로드: resume 이면 last.pt, 신규이면 base 모델
        if fold_resume:
            patch_grad_scaler(last_pt)
            model = YOLO(str(last_pt))
        else:
            model_path = PROJECT_ROOT / tc["model"]
            model = YOLO(str(model_path))

        model.add_callback("on_pretrain_routine_end", eta_callback.on_pretrain_routine_end)
        model.add_callback("on_train_batch_end", eta_callback.on_train_batch_end)
        model.add_callback("on_val_end", eta_callback.on_val_end)
        model.add_callback("on_train_end", eta_callback.on_train_end)

        results = model.train(
            data=str(data_yaml),
            epochs=tc["epochs"],
            batch=tc["batch"],
            imgsz=tc["imgsz"],
            workers=tc.get("workers", 4),
            cache=tc.get("cache", False),
            patience=tc.get("patience", 50),
            device=device,
            project=str(paths["runs"] / "kfold"),
            name=f"fold_{fold_num}",
            exist_ok=True,
            resume=fold_resume,
            verbose=False,
            optimizer=tc.get("optimizer", "auto"),
            lr0=tc.get("lr0", 0.01),
            lrf=tc.get("lrf", 0.01),
            cos_lr=tc.get("cos_lr", False),
            flipud=tc.get("flipud", 0.0),
            fliplr=tc.get("fliplr", 0.5),
            mosaic=tc.get("mosaic", 1.0),
            box=tc.get("box", 10.0),
            cls=tc.get("cls", 0.5),
            dfl=tc.get("dfl", 2.0),
            rect=tc.get("rect", True),
            iou=tc.get("iou", 0.7),
        )

        # best.pt 백업 (폴드 번호 1-indexed로 통일)
        best_src = Path(results.save_dir) / "weights" / "best.pt"
        best_dst = paths["weights"] / f"best_fold_{fold_num}.pt"
        if best_src.exists():
            shutil.copy(best_src, best_dst)
            print(f"[kfold] Fold {fold_num} 최적 가중치 저장 완료: {best_dst}")
        else:
            print(f"[warn] Fold {fold_num} 최적 가중치를 찾지 못했습니다: {best_src}")

        data_yaml.unlink(missing_ok=True)
        # 텍스트 파일은 디버깅/재현을 위해 남겨두는 것도 좋지만, 필요하다면 삭제 가능
        # train_txt.unlink(missing_ok=True)
        # val_txt.unlink(missing_ok=True)
        
    print(f"\n[kfold] 모든 {k}-Fold 학습이 완료되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume K-Fold from checkpoints. Skips completed folds, resumes interrupted folds from last.pt.",
    )
    args = parser.parse_args()
    main(args.config, args.resume)
