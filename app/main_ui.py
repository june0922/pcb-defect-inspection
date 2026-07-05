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
    QApplication, QAction, QShortcut, QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSlot
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QIcon, QKeySequence
)

from vision_viewer import VisionViewer, confidence_color
from inference_worker import InferenceWorker

# ── 프로젝트 루트 ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


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
    verdict: str = "pending"   # "pending" | "pass" | "1"~"6"


BORDER_COLORS = {
    "pending": QColor(128, 128, 128),
    "pass":    QColor(0, 200, 0),
    "1":       QColor(255, 50, 50),
    "2":       QColor(255, 50, 50),
    "3":       QColor(255, 50, 50),
    "4":       QColor(255, 50, 50),
    "5":       QColor(255, 50, 50),
    "6":       QColor(255, 50, 50),
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


# ── SettingsDialog ────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(300)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 100)
        self.min_spin.setSuffix(" %")
        self.min_spin.setValue(current_settings.get("min_conf", 50))
        
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 100)
        self.max_spin.setSuffix(" %")
        self.max_spin.setValue(current_settings.get("max_conf", 100))
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.10, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(current_settings.get("iou_thresh", 0.45))
        
        form_layout.addRow("Min Confidence:", self.min_spin)
        form_layout.addRow("Max Confidence:", self.max_spin)
        form_layout.addRow("IoU Threshold:", self.iou_spin)
        
        layout.addLayout(form_layout)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ccc; }
            QLabel { color: #ccc; }
            QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; color: #ccc; padding: 2px; }
            QPushButton { background-color: #333; color: #ccc; padding: 5px; }
            QPushButton:hover { background-color: #444; }
        """)

    def get_settings(self):
        return {
            "min_conf": int(self.min_spin.value()),
            "max_conf": int(self.max_spin.value()),
            "iou_thresh": round(float(self.iou_spin.value()), 2)
        }


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
    - F      → Fail (진성 불량)
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
        
        self._current_folder = None
        self._edited_images: dict[str, np.ndarray] = {}  # 편집된 이미지 캐시 (경로→ndarray)
        self._settings_file = Path(__file__).parent / "settings.json"
        self._app_settings = self._load_settings()

        self._init_ui()
        self._init_menu()
        self._connect_signals()

        # 시작 시 빈 창으로 대기 (사용자가 직접 File > Open Folder 메뉴 이용)
        self._status_label.setText("Ready. File > Open Folder... to start.")

    def _load_settings(self):
        default_settings = {"min_conf": 40, "max_conf": 90, "iou_thresh": 0.45}
        if self._settings_file.exists():
            try:
                with open(self._settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    merged = {**default_settings, **settings}
                    return {
                        "min_conf": int(merged.get("min_conf", 40)),
                        "max_conf": int(merged.get("max_conf", 90)),
                        "iou_thresh": round(float(merged.get("iou_thresh", 0.45)), 2)
                    }
            except Exception:
                pass
        return default_settings

    def _save_settings(self, settings):
        try:
            with open(self._settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save settings: {e}")

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
        self._global_view.setFocusPolicy(Qt.NoFocus)
        self._global_pixmap_item = None
        self._crosshair_items: list = []
        self._global_cursor_item = None  # GlobalView 브러쉬 커서 아이템
        self._h_splitter.addWidget(self._global_view)

        # Local View (Top-Right, 70%)
        self._local_view = VisionViewer()
        self._local_view.setFocusPolicy(Qt.NoFocus)
        self._h_splitter.addWidget(self._local_view)

        self._h_splitter.setStretchFactor(0, 3)
        self._h_splitter.setStretchFactor(1, 7)

        # ── 하단: FilmStrip ──
        self._filmstrip = QListWidget()
        self._filmstrip.setFocusPolicy(Qt.NoFocus)
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

        option_menu = menu_bar.addMenu("Option")
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        option_menu.addAction(settings_action)

    def _open_settings_dialog(self):
        dialog = SettingsDialog(self._app_settings, self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            new_settings = dialog.get_settings()
            
            # 변경 여부를 명시적으로 확인
            is_changed = False
            for k in ["min_conf", "max_conf", "iou_thresh"]:
                if new_settings[k] != self._app_settings.get(k):
                    is_changed = True
                    break
                    
            if is_changed:
                if getattr(self, "_current_folder", None):
                    reply = QMessageBox.question(
                        self, "Settings Changed",
                        "옵션이 변경되었습니다. 수정된 이미지를 반영하여 다시 연산합니다.\n진행 중인 리뷰 내역은 초기화됩니다. 계속하시겠습니까?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self._app_settings = new_settings
                        self._save_settings(self._app_settings)
                        # UI 강제 갱신 후 즉시 재연산 (편집 이미지 보존)
                        QApplication.processEvents()
                        self._start_inference(self._current_folder, preserve_edits=True)
                else:
                    self._app_settings = new_settings
                    self._save_settings(self._app_settings)

    def _connect_signals(self):
        self._filmstrip.currentRowChanged.connect(self._on_filmstrip_selection)

        # 단축키 설정 (위젯 포커스 무관하게 전역 동작하도록 QShortcut 사용)
        self._shortcut_pass = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._shortcut_pass.setAutoRepeat(False)
        self._shortcut_pass.setContext(Qt.ApplicationShortcut)
        self._shortcut_pass.activated.connect(lambda: self._verdict_current("pass"))

        self._shortcuts_defect = []
        for i, key in enumerate([Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6], start=1):
            sc = QShortcut(QKeySequence(key), self)
            sc.setAutoRepeat(False)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda val=str(i): self._verdict_current(val))
            self._shortcuts_defect.append(sc)

        # 좌우 화살표 (이전/다음 결함 순차 이동)
        self._shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._shortcut_prev.setContext(Qt.ApplicationShortcut)
        self._shortcut_prev.activated.connect(self._navigate_previous)

        self._shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._shortcut_next.setContext(Qt.ApplicationShortcut)
        self._shortcut_next.activated.connect(self._navigate_next)

        # 상하 화살표는 스크롤 기본 동작을 방지하기 위해 무시 (아무 동작도 하지 않음)
        self._shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self._shortcut_up.activated.connect(lambda: None)

        self._shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self._shortcut_down.activated.connect(lambda: None)

        # W, A, S, D 패닝 (상하좌우 30픽셀씩 이동)
        PAN_STEP = 30
        self._shortcut_w = QShortcut(QKeySequence(Qt.Key_W), self)
        self._shortcut_w.activated.connect(lambda: self._local_view.pan(0, -PAN_STEP))

        self._shortcut_s = QShortcut(QKeySequence(Qt.Key_S), self)
        self._shortcut_s.activated.connect(lambda: self._local_view.pan(0, PAN_STEP))

        self._shortcut_a = QShortcut(QKeySequence(Qt.Key_A), self)
        self._shortcut_a.activated.connect(lambda: self._local_view.pan(-PAN_STEP, 0))

        self._shortcut_d = QShortcut(QKeySequence(Qt.Key_D), self)
        self._shortcut_d.activated.connect(lambda: self._local_view.pan(PAN_STEP, 0))

        # Q(축소), E(확대) 줌
        self._shortcut_q = QShortcut(QKeySequence(Qt.Key_Q), self)
        self._shortcut_q.activated.connect(lambda: self._local_view.zoom(zoom_in=False))

        self._shortcut_e = QShortcut(QKeySequence(Qt.Key_E), self)
        self._shortcut_e.activated.connect(lambda: self._local_view.zoom(zoom_in=True))

        # F5 → 현재 이미지 재연산
        self._shortcut_f5 = QShortcut(QKeySequence(Qt.Key_F5), self)
        self._shortcut_f5.setAutoRepeat(False)
        self._shortcut_f5.setContext(Qt.ApplicationShortcut)
        self._shortcut_f5.activated.connect(self._rerun_inference_current_image)

        # Ctrl+S → 결과 저장
        self._shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self._shortcut_save.setContext(Qt.ApplicationShortcut)
        self._shortcut_save.activated.connect(self._save_results)

        # 브러쉬 크기 조절 (- / = 키) — VisionViewer가 NoFocus이므로 MainWindow QShortcut 사용
        self._shortcut_brush_dec = QShortcut(QKeySequence(Qt.Key_Minus), self)
        self._shortcut_brush_dec.setContext(Qt.ApplicationShortcut)
        self._shortcut_brush_dec.activated.connect(self._decrease_brush_size)

        self._shortcut_brush_inc = QShortcut(QKeySequence(Qt.Key_Equal), self)
        self._shortcut_brush_inc.setContext(Qt.ApplicationShortcut)
        self._shortcut_brush_inc.activated.connect(self._increase_brush_size)

        # VisionViewer 시그널 구독
        self._local_view.image_edited.connect(self._on_image_edited)
        self._local_view.brush_size_changed.connect(self._on_brush_size_changed)
        self._local_view.cursor_moved.connect(self._on_cursor_moved)
        self._local_view.cursor_left.connect(self._on_cursor_left)

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

        # 이미 작업 중인 폴더가 있으면 확인 다이얼로그 표시
        if self._current_folder is not None:
            reply = QMessageBox.question(
                self, "Open New Folder",
                "새 폴더를 열면 진행 중인 리뷰 내역이 초기화됩니다.\n계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self._start_inference(folder)

    def _start_inference(self, folder, preserve_edits=False):
        self._current_folder = folder
        
        # 기존 워커 정리 로직 (기존 워커가 모델 로딩 등 작업 중일 경우 대비)
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            # blockSignals로 큐에 남은 시그널 차단 (disconnect보다 안전)
            self._worker.blockSignals(True)
            self._worker.wait(2000)
            self._worker = None

        # 기존 데이터 초기화 (편집 이미지는 preserve_edits에 따라 유지)
        self._defects.clear()
        self._filmstrip.clear()
        self._current_index = -1
        self._current_image = None
        self._current_image_path = None
        if not preserve_edits:
            self._edited_images.clear()
        self._global_scene.clear()
        self._global_pixmap_item = None
        self._global_cursor_item = None
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

        # 설정된 임계값 읽기
        min_conf = self._app_settings.get("min_conf", 50) / 100.0
        max_conf = self._app_settings.get("max_conf", 100) / 100.0
        iou_thresh = self._app_settings.get("iou_thresh", 0.45)

        # GPU/CPU 자동 판별
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        self._worker = InferenceWorker(
            weight_paths=weight_paths,
            min_conf=min_conf,
            max_conf=max_conf,
            iou_thresh=iou_thresh,
            device=device,
        )
        self._worker.set_image_paths(image_paths)
        self._worker.set_edited_images(self._edited_images)
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
                f"Space=Pass / 1~6=Fail / Shift=Hide overlay"
            )

    @pyqtSlot(str)
    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._status_label.setText(f"Error: {msg}")

    # ── 썸네일 생성/갱신 ───────────────────────────────────────

    def _create_thumbnail(self, crop_bgr, border_color, verdict="pending"):
        """BGR numpy 크롭 → 테두리 포함 QPixmap 변환."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pixmap)
        
        # 반투명 오버레이
        if verdict != "pending":
            overlay_color = QColor(border_color)
            overlay_color.setAlpha(64)
            painter.fillRect(0, 0, pixmap.width(), pixmap.height(), overlay_color)

        pen = QPen(border_color, BORDER_WIDTH)
        painter.setPen(pen)
        half = BORDER_WIDTH // 2
        painter.drawRect(
            half, half,
            pixmap.width() - BORDER_WIDTH,
            pixmap.height() - BORDER_WIDTH,
        )

        # 텍스트(숫자) 뱃지 추가 (PASS는 텍스트 생략)
        if verdict != "pending" and verdict != "pass":
            # 우측 상단에 뱃지 배경 그리기
            badge_size = 26
            badge_margin = 6
            x = pixmap.width() - badge_size - badge_margin
            y = badge_margin
            
            painter.setPen(Qt.NoPen)
            # 테두리 색상을 불투명하게 사용하여 뱃지 배경색으로 활용
            badge_bg = QColor(border_color)
            badge_bg.setAlpha(230)
            painter.setBrush(badge_bg)
            painter.drawRoundedRect(x, y, badge_size, badge_size, 6, 6)
            
            # 뱃지 중앙에 텍스트 그리기
            painter.setPen(QPen(Qt.white))
            font = painter.font()
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRectF(x, y, badge_size, badge_size), Qt.AlignCenter, verdict)

        painter.end()
        return pixmap

    def _update_thumbnail_border(self, index):
        """판정 결과에 따라 썸네일 테두리 색상 갱신."""
        if not (0 <= index < len(self._defects)):
            return
        entry = self._defects[index]
        color = BORDER_COLORS.get(entry.verdict, BORDER_COLORS["pending"])
        new_pixmap = self._create_thumbnail(entry.crop_bgr, color, entry.verdict)
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
            # 이전 이미지의 편집 내용을 캐시에 저장
            if self._current_image is not None and self._current_image_path is not None:
                self._edited_images[self._current_image_path] = self._current_image

            # 편집된 이미지가 캐시에 있으면 디스크 대신 캐시에서 로드
            if entry.image_path in self._edited_images:
                self._current_image = self._edited_images[entry.image_path]
            else:
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
        self._global_cursor_item = None
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
        failed = sum(1 for d in self._defects if d.verdict in [str(i) for i in range(1, 7)])
        current = self._current_index + 1 if self._current_index >= 0 else 0
        self._status_label.setText(
            f"[{current}/{total}]  "
            f"Pass: {passed} | Fail: {failed} | Pending: {total - reviewed}"
        )

    # ── 단축키 ─────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(False)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(True)
        else:
            super().keyReleaseEvent(event)

    def _verdict_current(self, verdict):
        """현재 결함에 판정(Pass/Fail) 적용.
        단, 미판정(pending) 상태였던 것을 판정할 때만 다음 결함으로 자동 이동하며,
        이미 판정된 결함의 마킹을 수정할 때는 그 자리에 머무릅니다.
        """
        now = time.time()
        if now - self._last_verdict_time < DEBOUNCE_SEC:
            return
        self._last_verdict_time = now

        if not (0 <= self._current_index < len(self._defects)):
            return

        entry = self._defects[self._current_index]
        
        # 변경 전 상태가 'pending'이었는지 확인
        was_pending = (entry.verdict == "pending")
        
        entry.verdict = verdict
        self._update_thumbnail_border(self._current_index)
        
        # 처음 마킹하는 경우에만 다음 미판정 결함으로 자동 이동
        if was_pending:
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

        # 모든 결함 리뷰 완료 시 팝업 띄우던 로직 제거 -> 사용자가 직접 Ctrl+S로 저장

    def _navigate_previous(self):
        """이전 결함으로 수동 이동 (래핑 안함)."""
        if not self._defects:
            return
        idx = self._current_index - 1
        if idx >= 0:
            self._filmstrip.setCurrentRow(idx)

    def _navigate_next(self):
        """다음 결함으로 수동 이동 (래핑 안함)."""
        if not self._defects:
            return
        idx = self._current_index + 1
        if idx < len(self._defects):
            self._filmstrip.setCurrentRow(idx)

    def _show_completion_summary(self):
        passed = sum(1 for d in self._defects if d.verdict == "pass")
        failed = sum(1 for d in self._defects if d.verdict in [str(i) for i in range(1, 7)])
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

    # ── 브러쉬 편집 시그널 처리 ─────────────────────────────────

    @pyqtSlot(object)
    def _on_image_edited(self, edited_image: np.ndarray):
        """VisionViewer에서 브러쉬 편집 완료 시 호출.

        편집된 이미지를 _current_image 및 캐시에 동기화하고,
        현재 보고 있는 결함의 썸네일 크롭도 갱신합니다.
        GlobalView도 실시간 반영합니다.
        """
        self._current_image = edited_image
        if self._current_image_path:
            self._edited_images[self._current_image_path] = edited_image

        # 현재 결함의 크롭 재계산 (편집 결과가 크롭에도 반영)
        if 0 <= self._current_index < len(self._defects):
            entry = self._defects[self._current_index]
            if self._worker:
                new_crop = self._worker._compute_crop(
                    edited_image, entry.detection["bbox_abs"]
                )
                entry.crop_bgr = new_crop
                self._update_thumbnail_border(self._current_index)

        # GlobalView 실시간 갱신
        self._refresh_global_pixmap()

    @pyqtSlot(int)
    def _on_brush_size_changed(self, size: int):
        """브러쉬 크기 변경 시 상태바에 표시."""
        self._status_label.setText(f"Brush size: {size}px")

    def _decrease_brush_size(self):
        """브러쉬 크기 감소 (- 키)."""
        view = self._local_view
        view.set_brush_size(view._brush_size - 5)
        self._status_label.setText(f"Brush size: {view._brush_size}px")

    def _increase_brush_size(self):
        """브러쉬 크기 증가 (= 키)."""
        view = self._local_view
        view.set_brush_size(view._brush_size + 5)
        self._status_label.setText(f"Brush size: {view._brush_size}px")

    @pyqtSlot(float, float)
    def _on_cursor_moved(self, scene_x: float, scene_y: float):
        """VisionViewer의 마우스 위치를 GlobalView에 반투명 원으로 동기화."""
        r = self._local_view._brush_size / 2.0
        # orphan 아이템 감지: scene.clear() 후 참조가 남아있으면 폐기
        if self._global_cursor_item is not None and self._global_cursor_item.scene() is None:
            self._global_cursor_item = None
        if self._global_cursor_item is not None:
            self._global_cursor_item.setRect(
                scene_x - r, scene_y - r, r * 2, r * 2
            )
        else:
            pen = QPen(QColor(0, 220, 255, 180), 1)
            pen.setCosmetic(True)
            brush = QBrush(QColor(0, 220, 255, 40))
            self._global_cursor_item = self._global_scene.addEllipse(
                scene_x - r, scene_y - r, r * 2, r * 2, pen, brush
            )
            self._global_cursor_item.setZValue(1000)

    @pyqtSlot()
    def _on_cursor_left(self):
        """마우스가 VisionViewer를 떠나면 GlobalView 커서 제거."""
        if self._global_cursor_item is not None:
            # scene이 None이면 이미 clear()로 제거된 아이템이므로 참조만 해제
            if self._global_cursor_item.scene() is not None:
                self._global_scene.removeItem(self._global_cursor_item)
            self._global_cursor_item = None

    def _refresh_global_pixmap(self):
        """편집 후 GlobalView의 배경 pixmap만 교체 (십자선 등은 유지)."""
        if self._current_image is None or self._global_pixmap_item is None:
            return
        rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        self._global_pixmap_item.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _rerun_inference_current_image(self):
        """F5 키 → 현재 이미지(편집 반영)를 재연산하여 결함 목록 갱신.

        기존 워커의 로드된 모델을 재사용하여 동기 추론을 수행합니다.
        해당 이미지의 기존 결함 항목을 제거하고 새 결과로 교체합니다.
        """
        if self._current_image is None or self._current_image_path is None:
            self._status_label.setText("No image loaded for re-inference.")
            return

        if not self._worker or not self._worker._models:
            self._status_label.setText("Models not loaded. Cannot re-infer.")
            return

        self._status_label.setText("Re-running inference on edited image...")
        QApplication.processEvents()

        target_path = self._current_image_path

        # 1. 해당 이미지의 기존 결함 인덱스 범위 파악
        first_idx = None
        last_idx = None
        for i, d in enumerate(self._defects):
            if d.image_path == target_path:
                if first_idx is None:
                    first_idx = i
                last_idx = i

        # 2. 동기 추론 수행
        result = self._worker.run_single_image_sync(self._current_image)
        new_detections = result["detections"]
        new_crops = result["crops"]

        # 3. 기존 결함 제거 및 새 결함 삽입
        insert_pos = first_idx if first_idx is not None else len(self._defects)
        remove_count = (last_idx - first_idx + 1) if first_idx is not None else 0

        # 기존 항목 제거 (뒤에서부터 제거하여 인덱스 안정성 확보)
        for i in range(remove_count):
            del self._defects[insert_pos]
            self._filmstrip.takeItem(insert_pos)

        # 새 결함 삽입
        for i, (det, crop_bgr) in enumerate(zip(new_detections, new_crops)):
            pixmap = self._create_thumbnail(crop_bgr, BORDER_COLORS["pending"])
            entry = DefectEntry(
                defect_id=0,  # 아래에서 재번호 매김
                image_path=target_path,
                detection=det,
                all_detections=new_detections,
                detection_index=i,
                crop_bgr=crop_bgr,
                crop_pixmap=pixmap,
            )
            self._defects.insert(insert_pos + i, entry)

            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(THUMB_SIZE + 8, THUMB_SIZE + 8))
            item.setToolTip(
                f"{det['class_name']} ({det['confidence']:.2f})\n"
                f"{Path(target_path).name}"
            )
            self._filmstrip.insertItem(insert_pos + i, item)

        # 4. defect_id 재번호 매김 (전체 순서 정합성 보장)
        for i, d in enumerate(self._defects):
            d.defect_id = i

        # 5. 뷰 갱신
        new_count = len(new_detections)
        if new_count > 0:
            self._filmstrip.setCurrentRow(insert_pos)
        elif len(self._defects) > 0:
            safe_idx = min(insert_pos, len(self._defects) - 1)
            self._filmstrip.setCurrentRow(safe_idx)
        else:
            self._current_index = -1
            self._local_view.clear_all()
            self._global_scene.clear()

        self._update_status()
        self._status_label.setText(
            f"Re-inference complete. {new_count} defects found on this image."
        )

    # ── 종료 처리 ──────────────────────────────────────────────

    def closeEvent(self, event):
        """앱 종료 시 Worker 스레드 안전 정리."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
