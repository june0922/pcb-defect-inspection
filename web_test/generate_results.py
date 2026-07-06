"""웹 뷰어를 위한 모델 추론 결과 생성 스크립트.

튜닝 이전 모델(K-Fold 5개 WBF 앙상블)과 튜닝 이후 모델(K-Fold 5개 WBF 앙상블)의 
test 세트 추론 결과를 이미지로 저장하고,
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

def run_wbf(models, img, conf_thres, iou_thres, img_w, img_h, class_names):
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
    notune_dir = results_dir / "patience15_old"
    yestune_dir = results_dir / "patience100_new"
    
    for d in [notune_dir, yestune_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] 모델 로딩 중...")
    
    notune_models = []
    for i in range(1, 6):
        m_path = PROJECT_ROOT / "weights" / "v8n_notune_721_5kfold_150epoch_15patience_old" / "v8n_notune_721_5kfold_150epoch" / "weights" / f"best_fold_{i}.pt"
        notune_models.append(YOLO(str(m_path)))

    yestune_models = []
    for i in range(1, 6):
        m_path = PROJECT_ROOT / "weights" / "v8n_notune_721_5kfold_300epoch_100patience" / "weights" / f"best_fold_{i}.pt"
        yestune_models.append(YOLO(str(m_path)))
        
    class_names = notune_models[0].names

    metric_notune = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric_yestune = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    conf_thres = cfg["judge"]["conf_threshold"]
    iou_thres = cfg["judge"]["iou_threshold"]

    print("\n[INFO] 추론 및 WBF 앙상블 적용 중...")
    
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
        
        # No-Tune Inference (Old)
        n_boxes_abs, n_scores, n_labels = run_wbf(notune_models, img, conf_thres, iou_thres, img_w, img_h, class_names)
        
        img_notune = draw_boxes(img.copy(), n_boxes_abs, n_scores, n_labels, class_names)
        cv2.imwrite(str(notune_dir / img_name), img_notune)
        
        pred_notune = [dict(
            boxes=torch.tensor(n_boxes_abs, dtype=torch.float32) if len(n_boxes_abs) else torch.empty((0, 4), dtype=torch.float32),
            scores=torch.tensor(n_scores, dtype=torch.float32) if len(n_scores) else torch.empty((0,), dtype=torch.float32),
            labels=torch.tensor(n_labels, dtype=torch.int64) if len(n_labels) else torch.empty((0,), dtype=torch.int64)
        )]
        metric_notune.update(pred_notune, target)
        
        # Yes-Tune Inference (New)
        y_boxes_abs, y_scores, y_labels = run_wbf(yestune_models, img, conf_thres, iou_thres, img_w, img_h, class_names)
        
        img_yestune = draw_boxes(img.copy(), y_boxes_abs, y_scores, y_labels, class_names)
        cv2.imwrite(str(yestune_dir / img_name), img_yestune)
        
        pred_yestune = [dict(
            boxes=torch.tensor(y_boxes_abs, dtype=torch.float32) if len(y_boxes_abs) else torch.empty((0, 4), dtype=torch.float32),
            scores=torch.tensor(y_scores, dtype=torch.float32) if len(y_scores) else torch.empty((0,), dtype=torch.float32),
            labels=torch.tensor(y_labels, dtype=torch.int64) if len(y_labels) else torch.empty((0,), dtype=torch.int64)
        )]
        metric_yestune.update(pred_yestune, target)

    print("\n[INFO] 평가지표(mAP) 계산 중... (수 분이 소요될 수 있습니다)")
    notune_res = metric_notune.compute()
    yestune_res = metric_yestune.compute()
    
    def m(val): return float(val.item()) if hasattr(val, 'item') else float(val)
    
    print("\n=== 15 Patience Ensemble Old (WBF) ===")
    print(f"Recall:    {m(notune_res['mar_100']):.4f}")
    print(f"mAP@0.5:   {m(notune_res['map_50']):.4f}")
    print(f"mAP@50-95: {m(notune_res['map']):.4f}")
    
    print("\n=== 100 Patience Ensemble New (WBF) ===")
    print(f"Recall:    {m(yestune_res['mar_100']):.4f}")
    print(f"mAP@0.5:   {m(yestune_res['map_50']):.4f}")
    print(f"mAP@50-95: {m(yestune_res['map']):.4f}")

    results_data = {
        "images": image_files,
        "metrics": {
            "notune": {
                "recall": m(notune_res['mar_100']),
                "map50": m(notune_res['map_50']),
                "map50_95": m(notune_res['map'])
            },
            "yestune": {
                "recall": m(yestune_res['mar_100']),
                "map50": m(yestune_res['map_50']),
                "map50_95": m(yestune_res['map'])
            }
        }
    }
    
    with open(results_dir / "data.js", "w", encoding="utf-8") as f:
        f.write("const RESULTS_DATA = ")
        json.dump(results_data, f, ensure_ascii=False, indent=2)
        f.write(";\n")
    
    print("\n[INFO] 완료되었습니다. data.js가 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    generate_predictions()
