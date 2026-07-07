"""웹 뷰어를 위한 모델 추론 결과 생성 스크립트.

다양한 모델 설정(단일/앙상블)의 test 세트 추론 결과를 이미지로 저장하고,
torchmetrics를 통해 mAP를 직접 계산하여 
웹 UI에서 사용할 메타데이터(data.js)를 생성합니다.
"""

import sys
import json
import shutil
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_config, get_paths

# ==========================================
# 평가할 모델 설정 (단일 모델 또는 앙상블)
# 배열에 있는 모델들이 웹 뷰어에서 순서대로 보여집니다.
# 단일 모델의 경우 weights 배열에 1개의 경로만 입력하세요.
# 다중 모델(앙상블)의 경우 weights 배열에 여러 개의 경로를 입력하면 자동 WBF 적용됩니다.
# ==========================================
MODELS_TO_EVALUATE = [
    {
        "id": "model_a",
        "title": "No-Tune 0-KFold (300 Epoch)",
        "tooltip": "단일 모델(1 Train) 추론 결과입니다.",
        "result_dir": "model_a_results",
        "weights": [
            PROJECT_ROOT / "weights" / "new_v8n_notune_721_0kfold_300epoch_100patience" / "weights" / "best.pt"
        ]
    },
    {
        "id": "model_b",
        "title": "Yes-Tune 0-KFold (300 Epoch)",
        "tooltip": "단일 모델(1 Train) 추론 결과입니다.",
        "result_dir": "model_b_results",
        "weights": [
            PROJECT_ROOT / "weights" / "new_v8n_yestune_721_0kfold_300epoch_100patience" / "weights" / "best.pt"
        ]
    }
]

# (참고) 기존 5-Fold 앙상블 설정 백업
# MODELS_TO_EVALUATE = [
#     {
#         "id": "model_a",
#         "title": "300 Epoch 100 Patience Ensemble (5-Folds)",
#         "tooltip": "300 Epoch, 100 Patience로 학습한 K-Fold 5개 모델의 WBF 앙상블 결과입니다.",
#         "result_dir": "model_a_results",
#         "weights": [
#             PROJECT_ROOT / "weights" / "v8n_notune_721_5kfold_300epoch_100patience" / "weights" / f"best_fold_{i}.pt" for i in range(1, 6)
#         ]
#     },
#     {
#         "id": "model_b",
#         "title": "500 Epoch 100 Patience Ensemble (5-Folds)",
#         "tooltip": "500 Epoch, 100 Patience로 학습한 K-Fold 5개 모델의 WBF 앙상블 결과입니다.",
#         "result_dir": "model_b_results",
#         "weights": [
#             PROJECT_ROOT / "weights" / f"best_fold_{i}.pt" for i in range(1, 6)
#         ]
#     }
# ]
# ==========================================

def yolo2xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]

def norm2xyxy(x1, y1, x2, y2, img_w, img_h):
    return [x1*img_w, y1*img_h, x2*img_w, y2*img_h]

def load_ground_truth(label_path, img_w, img_h):
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c, x, y, w, h = map(float, parts[:5])
                labels.append(int(c))
                boxes.append(yolo2xyxy(x, y, w, h, img_w, img_h))
    return boxes, labels

def draw_boxes(img, boxes, scores, labels, class_names):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cls_name = class_names[label] if label < len(class_names) else str(label)
        color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def run_inference(models, img, conf_thres, iou_thres, img_w, img_h, class_names):
    if len(models) == 1:
        # 단일 모델 추론
        res = models[0].predict(img, conf=conf_thres, iou=iou_thres, verbose=False, device='cpu')[0]
        if len(res.boxes) > 0:
            boxes_abs = res.boxes.xyxy.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()
            labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            boxes_abs, scores, labels = [], [], []
        return boxes_abs, scores, labels
    else:
        # WBF 앙상블 추론
        all_boxes_norm = []
        all_scores = []
        all_labels = []
        
        for m in models:
            res = m.predict(img, conf=conf_thres, iou=iou_thres, verbose=False, device='cpu')[0]
            if len(res.boxes) > 0:
                boxes_norm = res.boxes.xyxyn.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
                all_boxes_norm.append(boxes_norm)
                all_scores.append(scores)
                all_labels.append(labels)
            else:
                all_boxes_norm.append([])
                all_scores.append([])
                all_labels.append([])
                
        if any(len(b) > 0 for b in all_boxes_norm):
            boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
                all_boxes_norm, all_scores, all_labels, weights=None, iou_thr=iou_thres, skip_box_thr=conf_thres
            )
        else:
            boxes_wbf, scores_wbf, labels_wbf = [], [], []
            
        boxes_abs = [norm2xyxy(x1, y1, x2, y2, img_w, img_h) for x1, y1, x2, y2 in boxes_wbf]
        labels_out = [int(l) for l in labels_wbf]
        
        return boxes_abs, scores_wbf, labels_out

def generate_predictions():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    paths = get_paths(cfg)
    
    test_images_dir = paths["processed"] / "images" / "test"
    test_labels_dir = paths["processed"] / "labels" / "test"
    
    if not test_images_dir.exists():
        print(f"[ERROR] 테스트 이미지가 없습니다: {test_images_dir}")
        sys.exit(1)
        
    image_files = sorted([f.name for f in test_images_dir.glob("*.jpg")])
    print(f"[INFO] 테스트 이미지 {len(image_files)}장 로드 완료.")

    web_dir = PROJECT_ROOT / "web_test"
    results_dir = web_dir / "results"
    
    print("\n[INFO] 모델 로딩 중...")
    
    for cfg_model in MODELS_TO_EVALUATE:
        d = results_dir / cfg_model["result_dir"]
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        
        models_list = []
        for w_path in cfg_model["weights"]:
            if not Path(w_path).exists():
                print(f"[WARNING] Weights not found: {w_path}")
                continue
            models_list.append(YOLO(str(w_path)))
            
        if not models_list:
            print(f"[ERROR] Model {cfg_model['id']} failed to load any weights.")
            sys.exit(1)
            
        cfg_model["loaded_models"] = models_list
        cfg_model["metric_calculator"] = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        
    # Get class names from the first loaded model
    class_names = MODELS_TO_EVALUATE[0]["loaded_models"][0].names

    conf_thres = cfg["judge"]["conf_threshold"]
    iou_thres = cfg["judge"]["iou_threshold"]

    print("\n[INFO] 추론 중...")
    
    for img_name in tqdm(image_files, desc="Processing Test Images"):
        img_path = str(test_images_dir / img_name)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        label_path = test_labels_dir / img_name.replace(".jpg", ".txt")
        gt_boxes, gt_labels = load_ground_truth(label_path, img_w, img_h)
        target = [dict(
            boxes=torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty((0, 4), dtype=torch.float32),
            labels=torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty((0,), dtype=torch.int64)
        )]
        
        for cfg_model in MODELS_TO_EVALUATE:
            models = cfg_model["loaded_models"]
            b_abs, scores, labels = run_inference(models, img, conf_thres, iou_thres, img_w, img_h, class_names)
            
            img_drawn = draw_boxes(img.copy(), b_abs, scores, labels, class_names)
            cv2.imwrite(str(results_dir / cfg_model["result_dir"] / img_name), img_drawn)
            
            pred = [dict(
                boxes=torch.tensor(b_abs, dtype=torch.float32) if len(b_abs) else torch.empty((0, 4), dtype=torch.float32),
                scores=torch.tensor(scores, dtype=torch.float32) if len(scores) else torch.empty((0,), dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.int64) if len(labels) else torch.empty((0,), dtype=torch.int64)
            )]
            cfg_model["metric_calculator"].update(pred, target)

    print("\n[INFO] 평가지표(mAP) 계산 중... (수 분이 소요될 수 있습니다)")
    
    def m(val): return float(val.item()) if hasattr(val, 'item') else float(val)
    
    results_data = {
        "images": image_files,
        "models": []
    }
    
    for cfg_model in MODELS_TO_EVALUATE:
        res = cfg_model["metric_calculator"].compute()
        
        print(f"\n=== {cfg_model['title']} ===")
        print(f"Recall:    {m(res['mar_100']):.4f}")
        print(f"mAP@0.5:   {m(res['map_50']):.4f}")
        print(f"mAP@50-95: {m(res['map']):.4f}")
        
        results_data["models"].append({
            "id": cfg_model["id"],
            "title": cfg_model["title"],
            "tooltip": cfg_model["tooltip"],
            "result_dir": cfg_model["result_dir"],
            "metrics": {
                "recall": m(res['mar_100']),
                "map50": m(res['map_50']),
                "map50_95": m(res['map'])
            }
        })
    
    with open(results_dir / "data.js", "w", encoding="utf-8") as f:
        f.write("const RESULTS_DATA = ")
        json.dump(results_data, f, ensure_ascii=False, indent=2)
        f.write(";\n")
    
    print("\n[INFO] 완료되었습니다. data.js가 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    generate_predictions()
