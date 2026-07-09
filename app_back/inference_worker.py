# 5-Fold K-Fold YOLO26s 앙상블 추론 워커 (QThread 기반 비동기 처리)

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# WBF 앙상블 병합용 IoU (YOLO26s는 NMS-free라 model.predict의 iou는 무의미하지만,
# 서로 다른 모델의 박스를 같은 객체로 묶는 WBF 클러스터링 자체에는 필요해 내부 상수로 유지)
_WBF_IOU_THR = 0.45


class InferenceWorker(QThread):
    """5개 K-Fold 모델의 WBF 앙상블 추론을 백그라운드 스레드에서 수행.

    모든 추론 결과는 pyqtSignal을 통해 메인 UI 스레드에 안전하게 전달됩니다.
    QThread 내부에서 UI 위젯에 절대 직접 접근하지 않습니다.
    """

    models_loaded = pyqtSignal()
    inference_done = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    THUMB_SIZE = 96
    CROP_PAD_RATIO = 0.3

    def __init__(self, weight_paths,
                 min_conf: float = 0.30,
                 max_conf: float = 1.0,
                 device: str = "cpu",
                 parent=None):
        """
        min_conf: WBF 및 model.predict 하한 (DB settings의 global_floor)
        max_conf: 표시 상한 (기본 1.0 — 모든 검출 표시)
        """
        super().__init__(parent)
        self._weight_paths = [str(p) for p in weight_paths]
        self.min_conf = min_conf      # public — poll 중 DB 변경 감지 시 직접 업데이트
        self.max_conf = max_conf
        self._device = device
        self._image_paths = []
        self._models = []
        self._running = True
        self._class_names = {}
        self._edited_images: dict = {}  # 편집된 이미지 캐시 (경로→ndarray)

    def set_image_paths(self, paths):
        """추론 대상 이미지 경로 리스트 설정."""
        self._image_paths = [str(p) for p in paths]

    def set_edited_images(self, edited_images: dict):
        """편집된 이미지 캐시를 설정. 해당 경로의 이미지는 디스크 대신 캐시에서 로드."""
        self._edited_images = edited_images

    def stop(self):
        """워커 스레드 중단 요청."""
        self._running = False

    def run(self):
        """스레드 진입점. 모델 로드 → 이미지 순회 → 앙상블 추론."""
        try:
            self._load_models()
            self.models_loaded.emit()
        except Exception as e:
            self.error.emit(f"모델 로딩 실패: {e}")
            return

        try:
            self._process_images()
        except Exception as e:
            self.error.emit(f"추론 실패: {e}")

    def _load_models(self):
        """weight_paths(1~5개) .pt 가중치를 메모리에 로드."""
        self._models, self._class_names = self._load_models_from_paths(self._weight_paths)

    def _load_models_from_paths(self, weight_paths):
        """가중치 경로 리스트를 전부 로드해 (models, class_names)를 반환.

        로컬 리스트에 전부 성공적으로 채운 뒤에만 반환한다 — 중간에 하나라도
        실패하면 예외를 던지고 self._models는 전혀 건드리지 않으므로, 호출자가
        실패 시 기존에 로딩되어 있던 모델 상태를 그대로 유지할 수 있다.
        """
        from ultralytics import YOLO

        models = []
        for path in weight_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"가중치 파일 없음: {path}")
            model = YOLO(path)
            models.append(model)

        class_names = models[0].names if models else {}
        return models, class_names

    def set_weight_paths_and_reload(self, weight_paths):
        """검사 모델 목록을 교체하고 동기적으로 다시 로드한다.

        새 모델들이 전부 성공적으로 로드된 뒤에만 weight_paths/models/class_names를
        교체한다. 일부 파일이 삭제되는 등 실패 시 예외를 그대로 전파하며, 이전까지
        정상 동작하던 모델 상태는 그대로 유지된다 — 호출자(app_back 메인 UI의 폴링
        루프)가 이 예외를 잡아 경고를 띄우고 기존 모델로 계속 동작하게 한다.
        """
        models, class_names = self._load_models_from_paths(weight_paths)
        self._weight_paths = [str(p) for p in weight_paths]
        self._models = models
        self._class_names = class_names

    def _process_images(self):
        """이미지 리스트를 순회하며 앙상블 추론 수행."""
        total = len(self._image_paths)
        for idx, img_path in enumerate(self._image_paths):
            if not self._running:
                break

            # 편집된 이미지가 캐시에 있으면 디스크 대신 사용
            if img_path in self._edited_images:
                img = self._edited_images[img_path].copy()
            else:
                img = cv2.imread(img_path)
            if img is None:
                self.progress.emit(idx + 1, total)
                continue

            h, w = img.shape[:2]
            detections = self._ensemble_predict(img, w, h)

            # 각 결함의 썸네일 크롭 생성 (96×96, 메인 스레드에서 QPixmap 변환)
            crops = []
            for det in detections:
                crop = self._compute_crop(img, det["bbox_abs"])
                crops.append(crop)

            result = {
                "image_path": img_path,
                "image_width": w,
                "image_height": h,
                "detections": detections,
                "crops": crops,
            }
            self.inference_done.emit(result)
            self.progress.emit(idx + 1, total)

            # 원본 이미지 참조 해제 → GC 회수 유도
            del img

    def _ensemble_predict(self, img_source, img_w, img_h):
        """5개 모델 예측 → WBF 병합 → 최종 결함 리스트 반환.

        Args:
            img_source: 이미지 파일 경로(str) 또는 ndarray.
        """
        from ensemble_boxes import weighted_boxes_fusion

        all_boxes_norm = []
        all_scores = []
        all_labels = []

        for model in self._models:
            # YOLO26s는 NMS-free(end2end)라 iou 인자는 무시되므로 전달하지 않는다.
            res = model.predict(
                img_source,
                conf=self.min_conf,
                verbose=False,
                device=self._device,
            )[0]

            if len(res.boxes) > 0:
                boxes_norm = res.boxes.xyxyn.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                boxes_norm, scores, labels = [], [], []

            all_boxes_norm.append(boxes_norm)
            all_scores.append(scores)
            all_labels.append(labels)

        # WBF 병합
        if not any(len(b) > 0 for b in all_boxes_norm):
            return []

        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            all_boxes_norm,
            all_scores,
            all_labels,
            weights=None,
            iou_thr=_WBF_IOU_THR,
            skip_box_thr=self.min_conf,
        )

        detections = []
        for box_n, score, label in zip(boxes_wbf, scores_wbf, labels_wbf):
            conf_score = float(score)
            if not (self.min_conf <= conf_score <= self.max_conf):
                continue

            x1n, y1n, x2n, y2n = box_n
            cls_id = int(label)
            detections.append({
                "bbox_abs": [
                    float(x1n * img_w),
                    float(y1n * img_h),
                    float(x2n * img_w),
                    float(y2n * img_h),
                ],
                "bbox_norm": [float(x1n), float(y1n), float(x2n), float(y2n)],
                "class_id": cls_id,
                "class_name": self._class_names.get(cls_id, str(cls_id)),
                "confidence": conf_score,
            })

        return detections

    def run_single_image_sync(self, img: np.ndarray):
        """메모리 내 ndarray 이미지 1장에 대해 동기 앙상블 추론 수행.

        이미 로드된 모델을 재사용하므로 모델 로딩 지연이 없습니다.
        F5 재연산 등 메인 스레드에서 호출합니다.

        Returns:
            dict: detections, crops 포함. detections가 없으면 빈 리스트.
        """
        from ensemble_boxes import weighted_boxes_fusion

        h, w = img.shape[:2]

        all_boxes_norm = []
        all_scores = []
        all_labels = []

        for model in self._models:
            # YOLO26s는 NMS-free(end2end)라 iou 인자는 무시되므로 전달하지 않는다.
            res = model.predict(
                img,
                conf=self.min_conf,
                verbose=False,
                device=self._device,
            )[0]

            if len(res.boxes) > 0:
                boxes_norm = res.boxes.xyxyn.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                boxes_norm, scores, labels = [], [], []

            all_boxes_norm.append(boxes_norm)
            all_scores.append(scores)
            all_labels.append(labels)

        if not any(len(b) > 0 for b in all_boxes_norm):
            return {"detections": [], "crops": []}

        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            all_boxes_norm,
            all_scores,
            all_labels,
            weights=None,
            iou_thr=_WBF_IOU_THR,
            skip_box_thr=self.min_conf,
        )

        detections = []
        for box_n, score, label in zip(boxes_wbf, scores_wbf, labels_wbf):
            conf_score = float(score)
            if not (self.min_conf <= conf_score <= self.max_conf):
                continue

            x1n, y1n, x2n, y2n = box_n
            cls_id = int(label)
            detections.append({
                "bbox_abs": [
                    float(x1n * w),
                    float(y1n * h),
                    float(x2n * w),
                    float(y2n * h),
                ],
                "bbox_norm": [float(x1n), float(y1n), float(x2n), float(y2n)],
                "class_id": cls_id,
                "class_name": self._class_names.get(cls_id, str(cls_id)),
                "confidence": conf_score,
            })

        crops = []
        for det in detections:
            crop = self._compute_crop(img, det["bbox_abs"])
            crops.append(crop)

        return {"detections": detections, "crops": crops}

    def _compute_crop(self, img, bbox_abs):
        """결함 영역 + 패딩을 크롭하여 썸네일 크기로 리사이즈하되,
        비율을 유지하며 정사각형으로 패딩하여 왜곡을 방지합니다."""
        x1, y1, x2, y2 = bbox_abs
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * self.CROP_PAD_RATIO
        pad_y = bh * self.CROP_PAD_RATIO

        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros(
                (self.THUMB_SIZE, self.THUMB_SIZE, 3), dtype=np.uint8
            )

        # 찌그러짐 방지: 정사각형으로 패딩 추가
        ch, cw = crop.shape[:2]
        max_side = max(ch, cw)
        
        top = (max_side - ch) // 2
        bottom = max_side - ch - top
        left = (max_side - cw) // 2
        right = max_side - cw - left
        
        square_crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[128, 128, 128]
        )

        return cv2.resize(
            square_crop,
            (self.THUMB_SIZE, self.THUMB_SIZE),
            interpolation=cv2.INTER_AREA,
        )
