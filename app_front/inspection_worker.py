# PCB 타일 단위 자동 검사 워커 (서펜타인 스캔 + 5K-Fold WBF 앙상블)

import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class InspectionWorker(QThread):
    """5개 K-Fold 모델의 WBF 앙상블로 타일 단위 자동 검사 수행.

    서펜타인(ㄹ자) 패턴으로 타일을 순회하며,
    각 타일의 PASS/FAIL/REVIEW 판정을 자동으로 수행합니다.
    """

    # Signals
    models_loaded = pyqtSignal()
    image_started = pyqtSignal(int, str)      # image_index, image_filename
    tile_inspected = pyqtSignal(dict)          # tile result dict
    all_done = pyqtSignal()
    progress = pyqtSignal(int, int)            # current_tile_global, total_tiles_global
    error = pyqtSignal(str)

    TILE_SIZE = 640
    THUMB_SIZE = 96

    def __init__(self, weight_paths,
                 per_class_bands: dict,
                 iou_thresh: float = 0.45,
                 device: str = "cpu",
                 parent=None):
        """
        per_class_bands: {class_id: (review_min, review_max)} — 0.0~1.0 비율
            review_min 미만 → PASS, review_min 이상 → REVIEW, review_max 초과 → FAIL
        """
        super().__init__(parent)
        self._weight_paths = [str(p) for p in weight_paths]
        self._per_class_bands = per_class_bands
        # WBF 및 model.predict의 전역 최소 신뢰도 = 가장 낮은 review_min
        self._global_floor = min(
            (band[0] for band in per_class_bands.values()),
            default=0.30,
        )
        self._iou_thresh = iou_thresh
        self._device = device
        self._image_paths = []
        self._models = []
        self._class_names = {}
        self._running = True
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

    def set_image_paths(self, paths: list):
        """검사 대상 이미지 경로 리스트 설정."""
        self._image_paths = [str(p) for p in paths]

    def pause(self):
        """검사 일시정지."""
        self._pause_event.clear()

    def resume(self):
        """검사 재개."""
        self._pause_event.set()

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def stop(self):
        """워커 스레드 중단 요청."""
        self._running = False
        self._pause_event.set()  # Unblock if paused

    def run(self):
        """스레드 진입점. 모델 로드 → 이미지 순회 → 타일별 앙상블 추론."""
        try:
            self._load_models()
            self.models_loaded.emit()
        except Exception as e:
            self.error.emit(f"모델 로딩 실패: {e}")
            return
        try:
            self._process_all_images()
        except Exception as e:
            self.error.emit(f"검사 실패: {e}")

    def _load_models(self):
        """5개 K-Fold .pt 가중치를 메모리에 로드."""
        from ultralytics import YOLO
        self._models = []
        for path in self._weight_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"가중치 파일 없음: {path}")
            model = YOLO(path)
            self._models.append(model)
        if self._models:
            self._class_names = self._models[0].names

    def _process_all_images(self):
        """모든 이미지를 순회하며 타일별 검사 수행."""
        # Phase 1: Pre-scan to count total tiles
        total_tiles = 0
        valid_paths = []
        for img_path in self._image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            rows, cols = self.compute_grid(h, w)
            total_tiles += rows * cols
            valid_paths.append(img_path)
            del img

        if total_tiles == 0:
            self.all_done.emit()
            return

        global_tile_idx = 0

        # Phase 2: Process each image
        for img_idx, img_path in enumerate(valid_paths):
            if not self._running:
                break

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            rows, cols = self.compute_grid(h, w)

            # Determine scan direction
            reverse = (img_idx % 2 == 1)
            scan_order = self.serpentine_order(rows, cols, reverse_start=reverse)

            self.image_started.emit(img_idx, Path(img_path).name)

            for row, col in scan_order:
                # Pause check
                self._pause_event.wait()
                if not self._running:
                    break

                # Extract tile
                tile = self._extract_tile(img, row, col)

                # Ensure BGR for model
                if len(tile.shape) == 2:
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                else:
                    tile_bgr = tile

                # Inference
                t0 = time.perf_counter()
                detections = self._ensemble_predict(tile_bgr)
                inference_ms = (time.perf_counter() - t0) * 1000

                # Verdict
                verdict = self._classify_verdict(detections)

                # Max confidence
                max_conf = max((d["confidence"] for d in detections), default=0.0)

                # Thumbnail
                thumb = self._create_thumbnail(tile_bgr)

                # Emit result
                self.tile_inspected.emit({
                    "image_index": img_idx,
                    "image_file": Path(img_path).name,
                    "image_path": img_path,
                    "grid_row": row,
                    "grid_col": col,
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "scan_order": global_tile_idx,
                    "tile_bgr": tile_bgr,
                    "verdict": verdict,
                    "detections": detections,
                    "max_confidence": max_conf,
                    "inference_time_ms": inference_ms,
                    "thumb_bgr": thumb,
                })

                global_tile_idx += 1
                self.progress.emit(global_tile_idx, total_tiles)

            del img  # Free memory

        if self._running:
            self.all_done.emit()

    @staticmethod
    def serpentine_order(rows: int, cols: int, reverse_start: bool = False) -> list:
        """서펜타인(ㄹ자) 스캔 순서 생성.

        reverse_start=False: 위→아래 ㄹ자 (row 0부터, 첫 행은 좌→우)
        reverse_start=True:  아래→위 ㄹ자 (마지막 row부터, 첫 행은 좌→우)

        Returns: list of (row, col) tuples
        """
        order = []
        row_range = range(rows - 1, -1, -1) if reverse_start else range(rows)
        for i, row in enumerate(row_range):
            col_range = range(cols - 1, -1, -1) if (i % 2 == 1) else range(cols)
            for col in col_range:
                order.append((row, col))
        return order

    @staticmethod
    def compute_grid(img_h: int, img_w: int, tile_size: int = 640) -> tuple:
        """이미지 크기로부터 그리드 행/열 수 계산.
        Returns: (rows, cols)
        """
        rows = math.ceil(img_h / tile_size)
        cols = math.ceil(img_w / tile_size)
        return rows, cols

    def _extract_tile(self, img: np.ndarray, row: int, col: int) -> np.ndarray:
        """이미지에서 (row, col) 위치의 640×640 타일 추출.
        이미지 경계를 초과하는 영역은 흰색(255)으로 패딩.
        """
        h, w = img.shape[:2]
        y1 = row * self.TILE_SIZE
        x1 = col * self.TILE_SIZE
        y2 = min(y1 + self.TILE_SIZE, h)
        x2 = min(x1 + self.TILE_SIZE, w)

        tile = img[y1:y2, x1:x2].copy()

        # Pad with white if tile is smaller than TILE_SIZE
        tile_h, tile_w = tile.shape[:2]
        if tile_h < self.TILE_SIZE or tile_w < self.TILE_SIZE:
            if len(tile.shape) == 2:  # Grayscale
                padded = np.full((self.TILE_SIZE, self.TILE_SIZE), 255, dtype=np.uint8)
            else:  # Color
                padded = np.full((self.TILE_SIZE, self.TILE_SIZE, tile.shape[2]), 255, dtype=np.uint8)
            padded[:tile_h, :tile_w] = tile
            tile = padded

        return tile

    def _ensemble_predict(self, img: np.ndarray) -> list:
        """5개 모델 예측 → WBF 병합 → 결함 리스트 반환.
        global_floor (6개 클래스 review_min 최솟값)을 model.predict 기준으로 사용.
        """
        from ensemble_boxes import weighted_boxes_fusion

        h, w = img.shape[:2]
        all_boxes_norm = []
        all_scores = []
        all_labels = []

        for model in self._models:
            res = model.predict(
                img,
                conf=self._global_floor,
                iou=self._iou_thresh,
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
            return []

        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            all_boxes_norm,
            all_scores,
            all_labels,
            weights=None,
            iou_thr=self._iou_thresh,
            skip_box_thr=self._global_floor,
        )

        detections = []
        for box_n, score, label in zip(boxes_wbf, scores_wbf, labels_wbf):
            conf_score = float(score)
            cls_id = int(label)
            r_min, _ = self._per_class_bands.get(cls_id, (self._global_floor, 1.0))
            if conf_score < r_min:
                continue
            x1n, y1n, x2n, y2n = box_n
            detections.append({
                "bbox_abs": [
                    float(x1n * w), float(y1n * h),
                    float(x2n * w), float(y2n * h),
                ],
                "bbox_norm": [float(x1n), float(y1n), float(x2n), float(y2n)],
                "class_id": cls_id,
                "class_name": self._class_names.get(cls_id, str(cls_id)),
                "confidence": conf_score,
            })

        return detections

    def _classify_verdict(self, detections: list) -> str:
        """클래스별 REVIEW 밴드 기준으로 타일 판정.

        - 어느 검출이든 해당 클래스의 review_max 초과 → FAIL (최우선)
        - FAIL 없고 review_min 이상인 검출 존재 → REVIEW
        - 모두 review_min 미만이거나 검출 없음 → PASS
        """
        if not detections:
            return "PASS"

        has_fail = False
        has_review = False

        for det in detections:
            cls_id = det["class_id"]
            conf = det["confidence"]
            r_min, r_max = self._per_class_bands.get(cls_id, (self._global_floor, 0.70))
            if conf > r_max:
                has_fail = True
                break
            elif conf >= r_min:
                has_review = True

        if has_fail:
            return "FAIL"
        elif has_review:
            return "REVIEW"
        return "PASS"

    def _create_thumbnail(self, tile: np.ndarray) -> np.ndarray:
        """640×640 타일을 96×96 썸네일로 리사이즈."""
        return cv2.resize(tile, (self.THUMB_SIZE, self.THUMB_SIZE), interpolation=cv2.INTER_AREA)
