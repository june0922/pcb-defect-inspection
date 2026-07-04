# PCB 결함 리뷰 스테이션 메인 윈도우 (3단 분할 레이아웃 + 단축키 워크플로우)

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QStatusBar, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QApplication, QAction,
)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSlot
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QIcon,
)

from vision_viewer import VisionViewer, confidence_color
from inference_worker import InferenceWorker

# ── 프로젝트 루트 및 config 유틸리티 ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from utils import load_config


# ── 데이터 모델 ───────────────────────────────────────────────

@dataclass
class DefectEntry:
    """리뷰 대기 결함 1건. crop_bgr(96×96)만 메모리에 유지."""

    defect_id: int
    image_path: str
    detection: dict            # bbox_abs, class_id, class_name, confidence
    all_detections: list       # 같은 이미지의 전체 검출 리스트 (참조 공유)
    detection_index: int       # all_detections 내 인덱스
    crop_bgr: np.ndarray = field(default=None, repr=False)
    crop_pixmap: QPixmap = field(default=None, repr=False)
    verdict: str = "pending"   # "pending" | "pass" | "fail"


BORDER_COLORS = {
    "pending": QColor(128, 128, 128),
    "pass":    QColor(0, 200, 0),
    "fail":    QColor(255, 50, 50),
}

THUMB_SIZE = 96
BORDER_WIDTH = 3
DEBOUNCE_SEC = 0.10


# ── GlobalView (미니맵) ───────────────────────────────────────

class GlobalView(QGraphicsView):
    """PCB 전체 미니맵. 리사이즈 시 자동 fitInView."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and self.scene().sceneRect().isValid():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


# ── MainWindow ────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """DeepPCB 결함 리뷰 스테이션 메인 윈도우.

    레이아웃:
    ┌──────────────────────────────────┐
    │  GlobalView(30%)  │ LocalView(70%) │  ← 상단 80%
    ├──────────────────────────────────┤
    │         FilmStrip (가로 스크롤)    │  ← 하단 20%
    └──────────────────────────────────┘

    단축키:
    - Space  → Pass (양품/False Call)
    - Enter  → Fail (진성 불량)
    - Shift(Hold) → 오버레이 숨김
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepPCB Defect Review Station")
        self.setMinimumSize(1280, 800)

        self._defects: list[DefectEntry] = []
        self._current_index: int = -1
        self._current_image: np.ndarray | None = None
        self._current_image_path: str | None = None
        self._last_verdict_time: float = 0.0
        self._worker: InferenceWorker | None = None

        self._init_ui()
        self._init_menu()
        self._connect_signals()

        # 시작 시 폴더 선택 다이얼로그
        QApplication.processEvents()
        self._select_folder_and_start()

    # ── UI 초기화 ──────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 상하 분할
        self._v_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self._v_splitter)

        # ── 상단: 좌우 분할 ──
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self._h_splitter = QSplitter(Qt.Horizontal)
        top_layout.addWidget(self._h_splitter)
        self._v_splitter.addWidget(top_widget)

        # Global View (Top-Left, 30%)
        self._global_view = GlobalView()
        self._global_scene = QGraphicsScene()
        self._global_view.setScene(self._global_scene)
        self._global_view.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self._global_view.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self._global_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._global_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._global_view.setInteractive(False)
        self._global_pixmap_item = None
        self._crosshair_items: list = []
        self._h_splitter.addWidget(self._global_view)

        # Local View (Top-Right, 70%)
        self._local_view = VisionViewer()
        self._h_splitter.addWidget(self._local_view)

        self._h_splitter.setStretchFactor(0, 3)
        self._h_splitter.setStretchFactor(1, 7)

        # ── 하단: FilmStrip ──
        self._filmstrip = QListWidget()
        self._filmstrip.setViewMode(QListWidget.IconMode)
        self._filmstrip.setFlow(QListWidget.LeftToRight)
        self._filmstrip.setWrapping(False)
        self._filmstrip.setResizeMode(QListWidget.Adjust)
        self._filmstrip.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._filmstrip.setSpacing(4)
        self._filmstrip.setMinimumHeight(THUMB_SIZE + 30)
        self._filmstrip.setMaximumHeight(THUMB_SIZE + 40)
        self._filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._filmstrip.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
            }
            QListWidget::item { padding: 2px; }
            QListWidget::item:selected {
                background-color: #2a4a7a;
                border: 2px solid #5599ff;
            }
        """)
        self._v_splitter.addWidget(self._filmstrip)

        self._v_splitter.setStretchFactor(0, 8)
        self._v_splitter.setStretchFactor(1, 2)

        # ── Status Bar ──
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        # Dark Theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QSplitter::handle { background-color: #333; width: 3px; height: 3px; }
            QStatusBar { background-color: #1a1a1a; color: #ccc; }
            QProgressBar {
                text-align: center;
                background-color: #2a2a2a;
                border: 1px solid #444;
                color: #ccc;
            }
            QProgressBar::chunk { background-color: #3a7bd5; }
            QLabel { color: #ccc; }
        """)

    def _init_menu(self):
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
            "QMenuBar { background: #1a1a1a; color: #ccc; }"
            "QMenuBar::item:selected { background: #333; }"
            "QMenu { background: #2a2a2a; color: #ccc; }"
            "QMenu::item:selected { background: #3a7bd5; }"
        )
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._select_folder_and_start)
        file_menu.addAction(open_action)

        save_action = QAction("Save Results...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_results)
        file_menu.addAction(save_action)

    def _connect_signals(self):
        self._filmstrip.currentRowChanged.connect(self._on_filmstrip_selection)

    # ── 추론 시작 ──────────────────────────────────────────────

    def _select_folder_and_start(self):
        default_dir = str(_PROJECT_ROOT / "preprocessed_data" / "images" / "test")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder for Review", default_dir
        )
        if not folder:
            self._status_label.setText(
                "No folder selected. File > Open to start."
            )
            return
        self._start_inference(folder)

    def _start_inference(self, folder):
        # 기존 워커 정리
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

        # 기존 데이터 초기화
        self._defects.clear()
        self._filmstrip.clear()
        self._current_index = -1
        self._current_image = None
        self._current_image_path = None
        self._global_scene.clear()
        self._global_pixmap_item = None
        self._crosshair_items.clear()
        self._local_view.clear_all()

        folder_path = Path(folder)
        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not image_paths:
            QMessageBox.warning(
                self, "Warning", f"No images found in:\n{folder}"
            )
            return

        # 가중치 파일 경로 — weights/best_fold_{1-5}.pt
        weight_paths = []
        for i in range(1, 6):
            wp = _PROJECT_ROOT / "weights" / f"best_fold_{i}.pt"
            weight_paths.append(str(wp))

        # config 에서 임계값 읽기
        try:
            cfg = load_config(str(_PROJECT_ROOT / "config.yaml"))
            conf_thresh = cfg.get("judge", {}).get("conf_threshold", 0.5)
            iou_thresh = cfg.get("judge", {}).get("iou_threshold", 0.45)
        except Exception:
            conf_thresh, iou_thresh = 0.5, 0.45

        # GPU/CPU 자동 판별
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        self._worker = InferenceWorker(
            weight_paths=weight_paths,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device,
        )
        self._worker.set_image_paths(image_paths)
        self._worker.models_loaded.connect(self._on_models_loaded)
        self._worker.inference_done.connect(self._on_inference_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)

        self._status_label.setText("Loading 5 K-Fold models...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(image_paths))
        self._progress_bar.setValue(0)

        self._worker.start()

    # ── Worker Signal Slots ────────────────────────────────────

    @pyqtSlot()
    def _on_models_loaded(self):
        self._status_label.setText("Models loaded. Running inference...")

    @pyqtSlot(dict)
    def _on_inference_done(self, result):
        """추론 완료된 1장의 결과를 수신, 결함별 썸네일을 FilmStrip에 추가."""
        detections = result["detections"]
        crops = result["crops"]
        image_path = result["image_path"]

        if not detections:
            return

        for i, (det, crop_bgr) in enumerate(zip(detections, crops)):
            defect_id = len(self._defects)
            pixmap = self._create_thumbnail(crop_bgr, BORDER_COLORS["pending"])

            entry = DefectEntry(
                defect_id=defect_id,
                image_path=image_path,
                detection=det,
                all_detections=detections,
                detection_index=i,
                crop_bgr=crop_bgr,
                crop_pixmap=pixmap,
            )
            self._defects.append(entry)

            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(THUMB_SIZE + 8, THUMB_SIZE + 8))
            item.setToolTip(
                f"{det['class_name']} ({det['confidence']:.2f})\n"
                f"{Path(image_path).name}"
            )
            self._filmstrip.addItem(item)

        # 첫 결함 자동 선택
        if self._filmstrip.count() > 0 and self._current_index == -1:
            self._filmstrip.setCurrentRow(0)

    @pyqtSlot(int, int)
    def _on_progress(self, current, total):
        self._progress_bar.setValue(current)
        reviewed = sum(1 for d in self._defects if d.verdict != "pending")
        n_defects = len(self._defects)
        self._status_label.setText(
            f"Inference: {current}/{total} images | "
            f"Defects: {n_defects} found | Reviewed: {reviewed}/{n_defects}"
        )
        if current == total:
            self._progress_bar.setVisible(False)
            self._status_label.setText(
                f"Inference complete. {n_defects} defects found. "
                f"Space=Pass / Enter=Fail / Shift=Hide overlay"
            )

    @pyqtSlot(str)
    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._status_label.setText(f"Error: {msg}")

    # ── 썸네일 생성/갱신 ───────────────────────────────────────

    def _create_thumbnail(self, crop_bgr, border_color):
        """BGR numpy 크롭 → 테두리 포함 QPixmap 변환."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pixmap)
        pen = QPen(border_color, BORDER_WIDTH)
        painter.setPen(pen)
        half = BORDER_WIDTH // 2
        painter.drawRect(
            half, half,
            pixmap.width() - BORDER_WIDTH,
            pixmap.height() - BORDER_WIDTH,
        )
        painter.end()
        return pixmap

    def _update_thumbnail_border(self, index):
        """판정 결과에 따라 썸네일 테두리 색상 갱신."""
        if not (0 <= index < len(self._defects)):
            return
        entry = self._defects[index]
        color = BORDER_COLORS[entry.verdict]
        new_pixmap = self._create_thumbnail(entry.crop_bgr, color)
        entry.crop_pixmap = new_pixmap
        item = self._filmstrip.item(index)
        if item:
            item.setIcon(QIcon(new_pixmap))

    # ── 뷰 갱신 ───────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_filmstrip_selection(self, index):
        """FilmStrip 썸네일 선택 → GlobalView + LocalView 동기화."""
        if not (0 <= index < len(self._defects)):
            return

        self._current_index = index
        entry = self._defects[index]

        # 이미지 변경 시에만 재로드 (메모리 절약)
        if entry.image_path != self._current_image_path:
            # 이전 이미지 참조 해제
            self._current_image = None
            self._current_image = cv2.imread(entry.image_path)
            self._current_image_path = entry.image_path
            self._update_global_view()

        self._update_local_view(entry)
        self._update_crosshair(entry)
        self._update_status()

    def _update_global_view(self):
        """GlobalView에 현재 이미지 미니맵 표시."""
        if self._current_image is None:
            return

        self._global_scene.clear()
        self._global_pixmap_item = None
        self._crosshair_items = []

        rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        self._global_pixmap_item = self._global_scene.addPixmap(pixmap)
        self._global_scene.setSceneRect(QRectF(pixmap.rect()))
        self._global_view.fitInView(
            self._global_scene.sceneRect(), Qt.KeepAspectRatio
        )

    def _update_crosshair(self, entry):
        """GlobalView에 십자선(Crosshair) 마커 동기화."""
        for item in self._crosshair_items:
            self._global_scene.removeItem(item)
        self._crosshair_items = []

        det = entry.detection
        x1, y1, x2, y2 = det["bbox_abs"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 반투명 시안 십자선
        cross_pen = QPen(QColor(0, 255, 255, 180), 1)
        cross_pen.setCosmetic(True)
        cross_pen.setStyle(Qt.DashLine)

        rect = self._global_scene.sceneRect()
        h_line = self._global_scene.addLine(
            rect.left(), cy, rect.right(), cy, cross_pen
        )
        v_line = self._global_scene.addLine(
            cx, rect.top(), cx, rect.bottom(), cross_pen
        )
        self._crosshair_items = [h_line, v_line]

        # 결함 위치 사각 마커
        color = confidence_color(det["confidence"])
        marker_pen = QPen(color, 2)
        marker_pen.setCosmetic(True)
        marker_rect = self._global_scene.addRect(
            x1, y1, x2 - x1, y2 - y1, marker_pen
        )
        self._crosshair_items.append(marker_rect)

    def _update_local_view(self, entry):
        """LocalView에 선택된 결함 주변 확대 이미지 + 오버레이 표시."""
        if self._current_image is None:
            return

        self._local_view.set_image(self._current_image)
        self._local_view.set_detections(
            entry.all_detections,
            highlight_index=entry.detection_index,
        )

        det = entry.detection
        x1, y1, x2, y2 = det["bbox_abs"]
        self._local_view.zoom_to_rect(x1, y1, x2, y2, pad_ratio=1.0)

    def _update_status(self):
        reviewed = sum(1 for d in self._defects if d.verdict != "pending")
        total = len(self._defects)
        passed = sum(1 for d in self._defects if d.verdict == "pass")
        failed = sum(1 for d in self._defects if d.verdict == "fail")
        current = self._current_index + 1 if self._current_index >= 0 else 0
        self._status_label.setText(
            f"[{current}/{total}]  "
            f"Pass: {passed} | Fail: {failed} | Pending: {total - reviewed}"
        )

    # ── 단축키 ─────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._verdict_current("pass")
        elif event.key() == Qt.Key_Return and not event.isAutoRepeat():
            self._verdict_current("fail")
        elif event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(False)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(True)
        else:
            super().keyReleaseEvent(event)

    def _verdict_current(self, verdict):
        """현재 결함에 판정(Pass/Fail) 적용 후 다음 이동. 100ms 디바운스."""
        now = time.time()
        if now - self._last_verdict_time < DEBOUNCE_SEC:
            return
        self._last_verdict_time = now

        if not (0 <= self._current_index < len(self._defects)):
            return

        entry = self._defects[self._current_index]
        entry.verdict = verdict
        self._update_thumbnail_border(self._current_index)
        self._advance_to_next()

    def _advance_to_next(self):
        """다음 미판정(pending) 결함으로 자동 포커스 이동."""
        total = len(self._defects)
        start = self._current_index + 1

        # 순방향 탐색
        for i in range(start, total):
            if self._defects[i].verdict == "pending":
                self._filmstrip.setCurrentRow(i)
                return

        # 래핑: 처음부터 재탐색
        for i in range(0, start):
            if self._defects[i].verdict == "pending":
                self._filmstrip.setCurrentRow(i)
                return

        # 모든 결함 리뷰 완료
        self._show_completion_summary()

    def _show_completion_summary(self):
        passed = sum(1 for d in self._defects if d.verdict == "pass")
        failed = sum(1 for d in self._defects if d.verdict == "fail")
        total = len(self._defects)

        msg = (
            f"Review Complete!\n\n"
            f"Total defects: {total}\n"
            f"Pass (False Call): {passed}\n"
            f"Fail (True Defect): {failed}\n\n"
            f"Save results?"
        )
        reply = QMessageBox.question(
            self, "Review Complete", msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._save_results()

    # ── 결과 저장 ──────────────────────────────────────────────

    def _save_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Review Results", "review_results.json",
            "JSON Files (*.json)",
        )
        if not path:
            return

        results = []
        for d in self._defects:
            results.append({
                "image": d.image_path,
                "class_name": d.detection["class_name"],
                "class_id": d.detection["class_id"],
                "confidence": d.detection["confidence"],
                "bbox": d.detection["bbox_abs"],
                "verdict": d.verdict,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self._status_label.setText(f"Results saved: {path}")

    # ── 종료 처리 ──────────────────────────────────────────────

    def closeEvent(self, event):
        """앱 종료 시 Worker 스레드 안전 정리."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
