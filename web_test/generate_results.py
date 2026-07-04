"""웹 뷰어를 위한 모델 추론 결과 생성 스크립트.

기본 모델(yolov8n.pt)과 앙상블 모델(best_fold_1~5.pt)의 
test 세트(150장) 추론 결과를 이미지로 저장하고,
웹 UI에서 사용할 메타데이터(metrics.json, data.json)를 생성합니다.
"""

import sys
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_config, get_paths
from evaluate import build_data_yaml

def generate_predictions():
    # 1. 설정 로드
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    paths = get_paths(cfg)
    
    test_images_dir = paths["processed"] / "images" / "test"
    if not test_images_dir.exists():
        print(f"[ERROR] 테스트 이미지가 없습니다: {test_images_dir}")
        sys.exit(1)
        
    image_files = sorted([f.name for f in test_images_dir.glob("*.jpg")])
    if not image_files:
        print("[ERROR] 테스트 이미지 폴더가 비어있습니다.")
        sys.exit(1)
        
    print(f"[INFO] 테스트 이미지 {len(image_files)}장 로드 완료.")

    # 2. 저장 폴더 설정
    web_dir = PROJECT_ROOT / "web_test"
    results_dir = web_dir / "results"
    base_dir = results_dir / "baseline"
    ens_dir = results_dir / "ensemble"
    
    for d in [base_dir, ens_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    # 3. data.yaml 임시 생성
    data_yaml = build_data_yaml(paths["processed"])

    # 4. Baseline (yolov8n.pt) 추론 및 평가
    print("\n[INFO] Baseline 모델 (yolov8n.pt) 평가 및 추론 시작...")
    base_model = YOLO(str(PROJECT_ROOT / "weights" / "yolov8n.pt"))
    
    # 평가 (Metrics 추출)
    base_metrics = base_model.val(data=str(data_yaml), split="test", exist_ok=True, plots=False)
    
    # 이미지 생성
    for img_name in image_files:
        img_path = str(test_images_dir / img_name)
        # save=True 하면 runs/detect/predict 에 생김. 직접 저장하는게 깔끔함.
        res = base_model.predict(img_path, conf=cfg["judge"]["conf_threshold"], iou=cfg["judge"]["iou_threshold"])[0]
        res.save(filename=str(base_dir / img_name))

    # 5. Final 모델 (best.pt) 추론 및 평가
    print("\n[INFO] Final 모델 (best.pt) 평가 및 추론 시작...")
    eval_weights = cfg.get("evaluate", {}).get("weights", "weights/best.pt")
    target_weight = str(PROJECT_ROOT / eval_weights[0]) if isinstance(eval_weights, list) else str(PROJECT_ROOT / eval_weights)
    ens_model = YOLO(target_weight)
    
    # 평가 (Metrics 추출)
    ens_metrics = ens_model.val(data=str(data_yaml), split="test", exist_ok=True, plots=False)
    
    # 이미지 생성
    for img_name in image_files:
        img_path = str(test_images_dir / img_name)
        res = ens_model.predict(img_path, conf=cfg["judge"]["conf_threshold"], iou=cfg["judge"]["iou_threshold"])[0]
        res.save(filename=str(ens_dir / img_name))

    # 6. JS 파일 생성 (로컬 HTML에서 CORS 에러 없이 로드하기 위함)
    def get_metrics_dict(m):
        return {
            "recall": float(m.box.mr) if hasattr(m.box, "mr") else 0.0,
            "map50": float(m.box.map50) if hasattr(m.box, "map50") else 0.0,
            "map50_95": float(m.box.map) if hasattr(m.box, "map") else 0.0
        }
        
    results_data = {
        "images": image_files,
        "metrics": {
            "baseline": get_metrics_dict(base_metrics),
            "ensemble": get_metrics_dict(ens_metrics)
        }
    }
    
    # Write to data.js
    with open(results_dir / "data.js", "w", encoding="utf-8") as f:
        f.write("const RESULTS_DATA = ")
        json.dump(results_data, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    data_yaml.unlink(missing_ok=True)
    print("\n[INFO] 모든 시각화 및 평가 데이터 생성 완료!")

if __name__ == "__main__":
    generate_predictions()
