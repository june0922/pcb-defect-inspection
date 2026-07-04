"""웹 뷰어를 위한 모델 추론 결과 생성 스크립트.

기본 모델(best.pt)과 앙상블 모델(best_fold_1~5.pt WBF 앙상블)의 
test 세트(150장) 추론 결과를 이미지로 저장하고,
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

PROJECT_ROOT = Path(__file__).parent.parent
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
    base_dir = results_dir / "baseline"
    ens_dir = results_dir / "ensemble"
    
    for d in [base_dir, ens_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] 모델 로딩 중...")
    base_model = YOLO(str(PROJECT_ROOT / "weights" / "best.pt"))
    class_names = base_model.names
    
    ens_models = []
    for i in range(1, 6):
        m_path = PROJECT_ROOT / "weights" / f"best_fold_{i}.pt"
        ens_models.append(YOLO(str(m_path)))

    metric_base = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric_ens = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    conf_thres = cfg["judge"]["conf_threshold"]
    iou_thres = cfg["judge"]["iou_threshold"]

    print("\n[INFO] 추론 및 WBF 앙상블 적용 중...")
    
    for img_name in image_files:
        img_path = str(test_images_dir / img_name)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        label_path = test_labels_dir / img_name.replace(".jpg", ".txt")
        gt_boxes, gt_labels = load_ground_truth(label_path, img_w, img_h)
        target = [dict(
            boxes=torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty((0, 4), dtype=torch.float32),
            labels=torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty((0,), dtype=torch.int64)
        )]
        
        res_base = base_model.predict(img_path, conf=conf_thres, iou=iou_thres, verbose=False)[0]
        b_boxes_abs = res_base.boxes.xyxy.cpu().numpy()
        b_scores = res_base.boxes.conf.cpu().numpy()
        b_labels = res_base.boxes.cls.cpu().numpy().astype(int)
        
        img_base = draw_boxes(img.copy(), b_boxes_abs, b_scores, b_labels, class_names)
        cv2.imwrite(str(base_dir / img_name), img_base)
        
        pred_base = [dict(
            boxes=torch.tensor(b_boxes_abs, dtype=torch.float32) if len(b_boxes_abs) else torch.empty((0, 4), dtype=torch.float32),
            scores=torch.tensor(b_scores, dtype=torch.float32) if len(b_scores) else torch.empty((0,), dtype=torch.float32),
            labels=torch.tensor(b_labels, dtype=torch.int64) if len(b_labels) else torch.empty((0,), dtype=torch.int64)
        )]
        metric_base.update(pred_base, target)
        
        all_boxes_norm = []
        all_scores = []
        all_labels = []
        
        for m in ens_models:
            res = m.predict(img_path, conf=conf_thres, iou=iou_thres, verbose=False)[0]
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
            
        e_boxes_abs = [norm2xyxy(x1, y1, x2, y2, img_w, img_h) for x1, y1, x2, y2 in boxes_wbf]
        e_labels = [int(l) for l in labels_wbf]
        
        img_ens = draw_boxes(img.copy(), e_boxes_abs, scores_wbf, e_labels, class_names)
        cv2.imwrite(str(ens_dir / img_name), img_ens)
        
        pred_ens = [dict(
            boxes=torch.tensor(e_boxes_abs, dtype=torch.float32) if len(e_boxes_abs) else torch.empty((0, 4), dtype=torch.float32),
            scores=torch.tensor(scores_wbf, dtype=torch.float32) if len(scores_wbf) else torch.empty((0,), dtype=torch.float32),
            labels=torch.tensor(e_labels, dtype=torch.int64) if len(e_labels) else torch.empty((0,), dtype=torch.int64)
        )]
        metric_ens.update(pred_ens, target)

    print("\n[INFO] 평가지표(mAP) 계산 중... (수 분이 소요될 수 있습니다)")
    base_res = metric_base.compute()
    ens_res = metric_ens.compute()
    
    def m(val): return float(val.item()) if hasattr(val, 'item') else float(val)
    
    print("\n=== Baseline (best.pt) ===")
    print(f"Recall:    {m(base_res['mar_100']):.4f}")
    print(f"mAP@0.5:   {m(base_res['map_50']):.4f}")
    print(f"mAP@50-95: {m(base_res['map']):.4f}")
    
    print("\n=== Ensemble (WBF) ===")
    print(f"Recall:    {m(ens_res['mar_100']):.4f}")
    print(f"mAP@0.5:   {m(ens_res['map_50']):.4f}")
    print(f"mAP@50-95: {m(ens_res['map']):.4f}")

    results_data = {
        "images": image_files,
        "metrics": {
            "baseline": {
                "recall": m(base_res['mar_100']),
                "map50": m(base_res['map_50']),
                "map50_95": m(base_res['map'])
            },
            "ensemble": {
                "recall": m(ens_res['mar_100']),
                "map50": m(ens_res['map_50']),
                "map50_95": m(ens_res['map'])
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
