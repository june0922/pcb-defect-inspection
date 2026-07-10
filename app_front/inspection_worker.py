# PCB 타일 단위 자동 검사 워커 (서펜타인 스캔 + 5K-Fold WBF 앙상블)

import math
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# WBF 앙상블 병합용 IoU (YOLO26s는 NMS-free라 model.predict의 iou는 무의미하지만,
# 서로 다른 모델의 박스를 같은 객체로 묶는 WBF 클러스터링 자체에는 필요해 내부 상수로 유지)
_WBF_IOU_THR = 0.45

# LocalView 초록(PASS 수준) 박스가 표시될 수 있는 confidence 하한 — 클래스별 review_min과
# 무관하게 고정. 이보다 낮은 confidence는 모델이 애초에 반환하지 않는다(model.predict의 conf 하한).
_GREEN_FLOOR = 0.01

# ── 프로젝트 루트 (db 모듈 import용) ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── DB 연동 (실패해도 검사는 계속 진행) ──────────────
# PNG 인코딩 + DB 저장을 이 워커의 백그라운드 스레드에서 직접 수행한다 — 예전에는
# UI 스레드(main_ui.py의 tile_inspected 핸들러)가 매 타일마다 동기로 이 작업을 했는데,
# 그 인코딩+커밋 비용이 UI를 막아 REVIEW 타일이 몰릴 때 렉의 원인이 되었다.
try:
    from db.database import upsert_tile as _db_upsert
    _DB_ENABLED = True
except Exception as _e:
    print(f"[DB] 연동 비활성화: {_e}")
    _DB_ENABLED = False


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

    THUMB_SIZE = 96

    def __init__(self, weight_paths,
                 per_class_bands: dict,
                 tile_size: int = 640,
                 overlap_pct: int = 0,
                 device: str = "cpu",
                 parent=None):
        """
        per_class_bands: {class_name: (review_min, review_max)} — 0.0~1.0 비율
            review_min 미만 → PASS, review_min 이상 → REVIEW, review_max 초과 → FAIL
            활성 클래스 목록에 없는 이름의 검출은 판정에서 제외된다.
        """
        super().__init__(parent)
        self._weight_paths = [str(p) for p in weight_paths]
        # Options 값은 생성 시점에 확정되며 이 워커의 수명 동안 바뀌지 않는다 —
        # 값이 바뀌려면 반드시 MainWindow가 새 워커로 전체 재시작한다.
        self.per_class_bands = per_class_bands
        # WBF 및 model.predict의 전역 최소 신뢰도 = _GREEN_FLOOR와 활성 클래스 review_min 중 최솟값
        # (review_min을 _GREEN_FLOOR보다 낮게 설정해도 REVIEW/FAIL 판정에 필요한 검출이 누락되지 않도록 안전)
        self._global_floor = min([_GREEN_FLOOR] + [band[0] for band in per_class_bands.values()])
        self.tile_size = tile_size
        self.overlap_pct = overlap_pct
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
            rows, cols = self.compute_grid(h, w, tile_size=self.tile_size, overlap_pct=self.overlap_pct)
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
            rows, cols = self.compute_grid(h, w, tile_size=self.tile_size, overlap_pct=self.overlap_pct)

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

                # Ensure BGR for model — cv2.imread(IMREAD_UNCHANGED)는 환경/버전에 따라
                # 흑백 PNG를 (H,W)가 아니라 (H,W,1)로 디코딩할 수 있어 둘 다 확인한다.
                if tile.ndim == 2 or tile.shape[2] == 1:
                    gray = tile if tile.ndim == 2 else tile[:, :, 0]
                    tile_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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

                # DB 기록 — 타일 이미지(PNG BLOB) + REVIEW/FAIL 등급 결함(개별 등급 포함).
                # 이 백그라운드 스레드에서 수행해 UI 스레드는 더 이상 이 비용을 부담하지 않는다.
                # 타일 1건 저장 실패가 전체 검사를 중단시키면 안 되므로 로그만 남기고 계속 진행한다.
                if _DB_ENABLED:
                    try:
                        dets_for_db = [
                            {**d, "verdict": self._classify_detection_verdict(d)}
                            for d in detections
                        ]
                        dets_for_db = [d for d in dets_for_db if d["verdict"] != "PASS"]
                        _db_upsert(tile_bgr, verdict, img_path, row, col, dets_for_db)
                    except Exception as e:
                        print(f"[DB] 타일 기록 실패 (row={row},col={col}): {e}")

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
    def _compute_stride(tile_size: int, overlap_pct: int = 0) -> int:
        """타일 크기와 오버랩 비율(%)로부터 스캔 스트라이드(px) 계산."""
        return max(1, round(tile_size * (1 - overlap_pct / 100.0)))

    @staticmethod
    def compute_grid(img_h: int, img_w: int, tile_size: int = 640, overlap_pct: int = 0) -> tuple:
        """이미지 크기로부터 그리드 행/열 수 계산 (오버랩 스트라이드 기반).

        타일은 stride 간격으로 배치되며 클램프하지 않는다 — 경계를 넘는 마지막
        타일은 _extract_tile()의 흰색 패딩이 그대로 처리한다. overlap_pct=0이면
        기존 ceil(L/tile_size) 동작과 100% 동일하다.
        Returns: (rows, cols)
        """
        stride = InspectionWorker._compute_stride(tile_size, overlap_pct)

        def _count(length: int) -> int:
            if length <= tile_size:
                return 1
            return math.ceil((length - tile_size) / stride) + 1

        return _count(img_h), _count(img_w)

    def _extract_tile(self, img: np.ndarray, row: int, col: int) -> np.ndarray:
        """이미지에서 (row, col) 위치의 tile_size×tile_size 타일 추출(오버랩 스트라이드 반영).
        이미지 경계를 초과하는 영역은 흰색(255)으로 패딩.
        """
        h, w = img.shape[:2]
        stride = self._compute_stride(self.tile_size, self.overlap_pct)
        y1 = row * stride
        x1 = col * stride
        y2 = min(y1 + self.tile_size, h)
        x2 = min(x1 + self.tile_size, w)

        tile = img[y1:y2, x1:x2].copy()

        # Pad with white if tile is smaller than tile_size
        tile_h, tile_w = tile.shape[:2]
        if tile_h < self.tile_size or tile_w < self.tile_size:
            if len(tile.shape) == 2:  # Grayscale
                padded = np.full((self.tile_size, self.tile_size), 255, dtype=np.uint8)
            else:  # Color
                padded = np.full((self.tile_size, self.tile_size, tile.shape[2]), 255, dtype=np.uint8)
            padded[:tile_h, :tile_w] = tile
            tile = padded

        return tile

    def _ensemble_predict(self, img: np.ndarray) -> list:
        """앙상블 모델 예측 → WBF 병합 → 결함 리스트 반환.
        global_floor (활성 클래스 review_min 최솟값)을 model.predict 기준으로 사용.
        활성 클래스 목록에 없는 이름의 검출은 결과에서 제외한다.
        """
        from ensemble_boxes import weighted_boxes_fusion

        # 타일 처리 도중 Options가 갱신되어도 내적 일관성을 유지하도록 1회 스냅샷
        bands = self.per_class_bands
        global_floor = self._global_floor

        h, w = img.shape[:2]
        all_boxes_norm = []
        all_scores = []
        all_labels = []

        for model in self._models:
            # YOLO26s는 NMS-free(end2end)라 iou 인자는 무시되므로 전달하지 않는다.
            res = model.predict(
                img,
                conf=global_floor,
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
            iou_thr=_WBF_IOU_THR,
            skip_box_thr=global_floor,
        )

        detections = []
        for box_n, score, label in zip(boxes_wbf, scores_wbf, labels_wbf):
            conf_score = float(score)
            cls_id = int(label)
            class_name = self._class_names.get(cls_id, str(cls_id))
            band = bands.get(class_name)
            if band is None:
                continue  # 활성 클래스 목록에 없음 — 검토 대상 아님
            # review_min 미만인 검출도 남긴다 — LocalView가 초록(PASS 수준) 박스로 표시한다.
            x1n, y1n, x2n, y2n = box_n
            detections.append({
                "bbox_abs": [
                    float(x1n * w), float(y1n * h),
                    float(x2n * w), float(y2n * h),
                ],
                "bbox_norm": [float(x1n), float(y1n), float(x2n), float(y2n)],
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf_score,
            })

        return detections

    def _classify_detection_verdict(self, det: dict) -> str:
        """개별 결함 1건의 클래스별 REVIEW 밴드 기준 등급 (REVIEW/FAIL/PASS)."""
        r_min, r_max = self.per_class_bands.get(det["class_name"], (self._global_floor, 0.70))
        conf = det["confidence"]
        if conf > r_max:
            return "FAIL"
        elif conf >= r_min:
            return "REVIEW"
        return "PASS"

    def _classify_verdict(self, detections: list) -> str:
        """클래스별 REVIEW 밴드 기준으로 타일 판정.

        - 어느 검출이든 해당 클래스의 review_max 초과 → FAIL (최우선)
        - FAIL 없고 review_min 이상인 검출 존재 → REVIEW
        - 모두 review_min 미만이거나 검출 없음 → PASS
        """
        if not detections:
            return "PASS"

        # 타일 처리 도중 Options가 갱신되어도 내적 일관성을 유지하도록 1회 스냅샷
        bands = self.per_class_bands
        global_floor = self._global_floor

        has_fail = False
        has_review = False

        for det in detections:
            conf = det["confidence"]
            r_min, r_max = bands.get(det["class_name"], (global_floor, 0.70))
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
        """타일을 96×96 썸네일로 리사이즈."""
        return cv2.resize(tile, (self.THUMB_SIZE, self.THUMB_SIZE), interpolation=cv2.INTER_AREA)
