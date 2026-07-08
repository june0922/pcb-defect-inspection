# PCB 결함 리뷰 스테이션 — DB에서 REVIEW 타일을 실시간으로 수신하여 추론 표시

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QStatusBar, QLabel, QMessageBox, QProgressBar,
    QApplication, QShortcut,
)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSlot, QTimer
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QIcon, QKeySequence,
)

from vision_viewer import VisionViewer, confidence_color
from inference_worker import InferenceWorker

# ── 프로젝트 루트 ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── DB 연동 ────────────────────────────────
try:
    from db.database import (
        init_db as _db_init,
        fetch_review_tiles as _db_fetch,
        get_settings as _db_get_settings,
        save_user_verdict as _db_save_verdict,
    )
    _DB_ENABLED = True
except Exception as _e:
    print(f"[DB] 연동 비활성화: {_e}")
    _DB_ENABLED = False

# 결함 클래스 이름 (데이터셋 기준)
_DEFECT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]

THUMB_SIZE = 96
BORDER_WIDTH = 3
DEBOUNCE_SEC = 0.10

BORDER_COLORS = {
    "pending": QColor(128, 128, 128),
    "pass":    QColor(0, 200, 0),
    "fail":    QColor(255, 50, 50),
}


# ── TileEntry ─────────────────────────────────────────────────
class TileEntry:
    """리뷰 대기 타일 1건."""
    def __init__(self, tile_id: int, img_bgr: np.ndarray,
                 detections: list, crops: list):
        self.tile_id = tile_id
        self.img_bgr = img_bgr
        self.detections = detections
        self.crops = crops
        self.verdict = "pending"


# ── MainWindow ────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """REVIEW 타일 실시간 수신 + 추론 표시 리뷰 스테이션.

    단축키:
    - Space → Pass (양품/오탐)
    - 1~6   → Fail (결함 클래스)
    - ←/→   → 이전/다음 타일 이동
    - Shift  → 오버레이 숨김 (Hold)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepPCB Review Station")
        self.setMinimumSize(1280, 800)

        self._tiles: list[TileEntry] = []
        self._current_index: int = -1
        self._last_verdict_time: float = 0.0
        self._last_shown_id: int = 0
        self._last_session_id: str = ""
        self._worker: InferenceWorker | None = None

        # DB에서 읽어온 현재 설정 (폴링마다 갱신)
        self._per_class_bands: dict = {i: (0.30, 0.70) for i in range(6)}
        self._iou_thresh: float = 0.45

        self._db_poll_timer = QTimer(self)
        self._db_poll_timer.setInterval(3000)
        self._db_poll_timer.timeout.connect(self._poll_db)

        self._init_ui()
        self._connect_signals()
        self._status_label.setText("모델 로딩 중... 잠시 기다려 주세요.")

        # 앱 시작 직후 모델 로딩 → 완료 후 DB 폴링 시작
        QTimer.singleShot(0, self._start_model_loading)

    # ── UI 초기화 ──────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        v_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(v_splitter)

        # 상단: 좌(GlobalView) / 우(LocalView)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        h_splitter = QSplitter(Qt.Horizontal)
        top_layout.addWidget(h_splitter)
        v_splitter.addWidget(top_widget)

        # GlobalView — 현재 타일 전체 미리보기
        self._global_view = QGraphicsView()
        self._global_scene = QGraphicsScene()
        self._global_view.setScene(self._global_scene)
        self._global_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._global_view.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self._global_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._global_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._global_view.setInteractive(False)
        h_splitter.addWidget(self._global_view)

        # LocalView — 결함 확대뷰
        self._local_view = VisionViewer()
        self._local_view.setFocusPolicy(Qt.NoFocus)
        h_splitter.addWidget(self._local_view)

        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 7)

        # 하단: FilmStrip
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
            QListWidget { background-color: #1a1a1a; border: 1px solid #333; }
            QListWidget::item { padding: 2px; }
            QListWidget::item:selected { background-color: #2a4a7a; border: 2px solid #5599ff; }
        """)
        v_splitter.addWidget(self._filmstrip)

        v_splitter.setStretchFactor(0, 8)
        v_splitter.setStretchFactor(1, 2)

        # StatusBar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QSplitter::handle { background-color: #333; width: 3px; height: 3px; }
            QStatusBar { background-color: #1a1a1a; color: #ccc; }
            QProgressBar { text-align: center; background-color: #2a2a2a; border: 1px solid #444; color: #ccc; }
            QProgressBar::chunk { background-color: #3a7bd5; }
            QLabel { color: #ccc; }
        """)

    def _connect_signals(self):
        self._filmstrip.currentRowChanged.connect(self._on_filmstrip_selection)

        self._sc_pass = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._sc_pass.setAutoRepeat(False)
        self._sc_pass.setContext(Qt.ApplicationShortcut)
        self._sc_pass.activated.connect(lambda: self._verdict_current("pass"))

        self._sc_defects = []
        for i, key in enumerate([Qt.Key_1, Qt.Key_2, Qt.Key_3,
                                   Qt.Key_4, Qt.Key_5, Qt.Key_6], start=1):
            sc = QShortcut(QKeySequence(key), self)
            sc.setAutoRepeat(False)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda val=str(i): self._verdict_current(val))
            self._sc_defects.append(sc)

        self._sc_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._sc_prev.setContext(Qt.ApplicationShortcut)
        self._sc_prev.activated.connect(self._navigate_previous)

        self._sc_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._sc_next.setContext(Qt.ApplicationShortcut)
        self._sc_next.activated.connect(self._navigate_next)

        # W/A/S/D 패닝
        PAN = 30
        for key, dx, dy in [(Qt.Key_W, 0, -PAN), (Qt.Key_S, 0, PAN),
                             (Qt.Key_A, -PAN, 0), (Qt.Key_D, PAN, 0)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda _dx=dx, _dy=dy: self._local_view.pan(_dx, _dy))

        sc_q = QShortcut(QKeySequence(Qt.Key_Q), self)
        sc_q.activated.connect(lambda: self._local_view.zoom(zoom_in=False))
        sc_e = QShortcut(QKeySequence(Qt.Key_E), self)
        sc_e.activated.connect(lambda: self._local_view.zoom(zoom_in=True))

    # ── 모델 로딩 ──────────────────────────────────────────────

    def _parse_bands(self, s: dict) -> dict:
        """DB settings dict → {class_id: (review_min, review_max)} (0.0~1.0 비율)."""
        bands = {}
        for i, cls in enumerate(_DEFECT_CLASSES):
            r_min = int(s.get(f"review_min_{cls}", 30)) / 100.0
            r_max = int(s.get(f"review_max_{cls}", 70)) / 100.0
            bands[i] = (r_min, r_max)
        return bands

    def _start_model_loading(self):
        weight_paths = [
            str(_PROJECT_ROOT / "weights" / f"best_fold_{i}.pt")
            for i in range(1, 6)
        ]
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        # DB에서 초기 설정 읽기 (실패 시 기본값 유지)
        if _DB_ENABLED:
            try:
                _db_init()
                s = _db_get_settings()
                self._per_class_bands = self._parse_bands(s)
                self._iou_thresh = float(s.get("iou_threshold", 0.45))
                self._last_session_id = s.get("db_session_id", "")
            except Exception as e:
                print(f"[DB] 초기 설정 로드 실패: {e}")

        global_floor = min(b[0] for b in self._per_class_bands.values())

        self._worker = InferenceWorker(
            weight_paths=weight_paths,
            min_conf=global_floor,
            max_conf=1.0,
            iou_thresh=self._iou_thresh,
            device=device,
        )
        self._worker.models_loaded.connect(self._on_models_loaded)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    @pyqtSlot()
    def _on_models_loaded(self):
        if _DB_ENABLED:
            self._db_poll_timer.start()
            self._status_label.setText(
                "[DB 연결됨] REVIEW 타일 대기 중... (3초마다 갱신)"
            )
        else:
            self._status_label.setText("모델 로딩 완료. (DB 비활성화)")

    @pyqtSlot(str)
    def _on_worker_error(self, msg):
        QMessageBox.critical(self, "모델 로딩 오류", msg)
        self._status_label.setText(f"오류: {msg}")

    # ── DB 폴링 ────────────────────────────────────────────────

    def _poll_db(self):
        """3초마다 DB 설정 동기화 + 새 REVIEW 타일 조회 → 추론 → FilmStrip 추가."""
        if not _DB_ENABLED or self._worker is None:
            return

        # ── 설정 동기화 ──────────────────────────────────────────
        try:
            s = _db_get_settings()
        except Exception as e:
            self._status_label.setText(f"[DB 폴링 오류] {e}")
            return

        # DB 초기화/세션 변경 감지
        current_session = s.get("db_session_id", "")
        if current_session != self._last_session_id:
            self._last_session_id = current_session
            self._on_db_reset()
            return  # 이번 폴링은 리셋 처리만

        # per-class bands 및 IoU 갱신
        new_bands = self._parse_bands(s)
        new_iou = float(s.get("iou_threshold", 0.45))
        bands_changed = new_bands != self._per_class_bands or new_iou != self._iou_thresh
        self._per_class_bands = new_bands
        self._iou_thresh = new_iou
        if bands_changed and self._worker:
            self._worker.min_conf = min(b[0] for b in new_bands.values())
            self._worker.iou_thresh = new_iou

        # ── 새 타일 조회 ─────────────────────────────────────────
        try:
            new_tiles = _db_fetch(after_id=self._last_shown_id)
        except Exception as e:
            self._status_label.setText(f"[DB 조회 오류] {e}")
            return

        for tile_row in new_tiles:
            self._last_shown_id = tile_row["id"]
            self._process_tile(tile_row)

        pending = sum(1 for t in self._tiles if t.verdict == "pending")
        total = len(self._tiles)
        self._status_label.setText(
            f"[DB 연결됨] 타일 {total}건 수신 | 미검토 {pending}건"
        )

    def _on_db_reset(self):
        """DB 초기화 감지 시 FilmStrip/상태 전체 리셋."""
        self._tiles.clear()
        self._filmstrip.clear()
        self._last_shown_id = 0
        self._current_index = -1
        self._local_view.clear_all()
        self._global_scene.clear()
        self._status_label.setText("DB가 초기화되었습니다. 새 검사를 기다리는 중...")

    def _process_tile(self, tile_row: dict):
        """PNG BLOB 디코딩 → 추론 → 타일 엔트리 생성 → FilmStrip 추가."""
        buf = np.frombuffer(tile_row["tile_image"], dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return

        # 메인 스레드에서 동기 추론 (640×640 타일 1장)
        result = self._worker.run_single_image_sync(img_bgr)
        detections = result.get("detections", [])
        crops = result.get("crops", [])

        entry = TileEntry(
            tile_id=tile_row["id"],
            img_bgr=img_bgr,
            detections=detections,
            crops=crops,
        )
        self._tiles.append(entry)

        # 썸네일 생성 (첫 번째 crop 또는 타일 전체 축소)
        if crops:
            thumb_bgr = cv2.resize(crops[0], (THUMB_SIZE, THUMB_SIZE))
        else:
            thumb_bgr = cv2.resize(img_bgr, (THUMB_SIZE, THUMB_SIZE))
        pixmap = self._make_thumbnail(thumb_bgr, BORDER_COLORS["pending"])

        item = QListWidgetItem()
        item.setIcon(QIcon(pixmap))
        item.setSizeHint(QSize(THUMB_SIZE + 8, THUMB_SIZE + 8))
        conf = max((d["confidence"] for d in detections), default=0.0)
        item.setToolTip(
            f"[REVIEW] Tile #{tile_row['id']}\n"
            f"검출: {len(detections)}건  최대 신뢰도: {conf:.2f}"
        )
        self._filmstrip.addItem(item)

        if self._current_index == -1:
            self._filmstrip.setCurrentRow(0)

    # ── 썸네일 ─────────────────────────────────────────────────

    def _make_thumbnail(self, bgr: np.ndarray, border_color: QColor,
                        verdict: str = "pending") -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pixmap)
        if verdict != "pending":
            overlay = QColor(border_color)
            overlay.setAlpha(64)
            painter.fillRect(0, 0, pixmap.width(), pixmap.height(), overlay)
        pen = QPen(border_color, BORDER_WIDTH)
        painter.setPen(pen)
        half = BORDER_WIDTH // 2
        painter.drawRect(half, half,
                         pixmap.width() - BORDER_WIDTH,
                         pixmap.height() - BORDER_WIDTH)
        if verdict not in ("pending", "pass"):
            badge = 26
            bm = 6
            bx = pixmap.width() - badge - bm
            by = bm
            painter.setPen(Qt.NoPen)
            bg = QColor(border_color)
            bg.setAlpha(230)
            painter.setBrush(bg)
            painter.drawRoundedRect(bx, by, badge, badge, 6, 6)
            painter.setPen(QPen(Qt.white))
            f = painter.font()
            f.setPointSize(14)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(QRectF(bx, by, badge, badge), Qt.AlignCenter, verdict)
        painter.end()
        return pixmap

    def _update_thumbnail(self, index: int):
        if not (0 <= index < len(self._tiles)):
            return
        entry = self._tiles[index]
        color = BORDER_COLORS.get(
            entry.verdict if entry.verdict in BORDER_COLORS else "fail",
            BORDER_COLORS["pending"]
        )
        if entry.crops:
            thumb = cv2.resize(entry.crops[0], (THUMB_SIZE, THUMB_SIZE))
        else:
            thumb = cv2.resize(entry.img_bgr, (THUMB_SIZE, THUMB_SIZE))
        pixmap = self._make_thumbnail(thumb, color, entry.verdict)
        item = self._filmstrip.item(index)
        if item:
            item.setIcon(QIcon(pixmap))

    # ── 뷰 갱신 ───────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_filmstrip_selection(self, index: int):
        if not (0 <= index < len(self._tiles)):
            return
        self._current_index = index
        entry = self._tiles[index]

        # GlobalView — 타일 이미지 전체 미리보기
        self._global_scene.clear()
        rgb = cv2.cvtColor(entry.img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self._global_scene.addPixmap(pixmap)
        self._global_scene.setSceneRect(QRectF(pixmap.rect()))
        self._global_view.fitInView(
            self._global_scene.sceneRect(), Qt.KeepAspectRatio
        )

        # LocalView — 결함 확대뷰 (첫 번째 결함 위치로 줌)
        self._local_view.set_image(entry.img_bgr)
        self._local_view.set_detections(
            entry.detections,
            highlight_index=0 if entry.detections else -1,
            per_class_bands=self._per_class_bands,
        )
        if entry.detections:
            x1, y1, x2, y2 = entry.detections[0]["bbox_abs"]
            self._local_view.zoom_to_rect(x1, y1, x2, y2, pad_ratio=1.0)

        self._update_status()

    def _update_status(self):
        reviewed = sum(1 for t in self._tiles if t.verdict != "pending")
        total = len(self._tiles)
        passed = sum(1 for t in self._tiles if t.verdict == "pass")
        failed = sum(1 for t in self._tiles if t.verdict not in ("pending", "pass"))
        cur = self._current_index + 1 if self._current_index >= 0 else 0
        self._status_label.setText(
            f"[{cur}/{total}]  Pass: {passed} | Fail: {failed} | 미검토: {total - reviewed}"
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

    def _verdict_current(self, verdict: str):
        now = time.time()
        if now - self._last_verdict_time < DEBOUNCE_SEC:
            return
        self._last_verdict_time = now
        if not (0 <= self._current_index < len(self._tiles)):
            return

        entry = self._tiles[self._current_index]
        was_pending = entry.verdict == "pending"
        entry.verdict = verdict

        # DB에 사용자 판정 저장
        if _DB_ENABLED:
            try:
                _db_save_verdict(entry.tile_id, verdict)
            except Exception as e:
                print(f"[DB] 판정 저장 실패 (tile_id={entry.tile_id}): {e}")

        self._update_thumbnail(self._current_index)
        if was_pending:
            self._advance_to_next()

    def _advance_to_next(self):
        for i in range(self._current_index + 1, len(self._tiles)):
            if self._tiles[i].verdict == "pending":
                self._filmstrip.setCurrentRow(i)
                return

    def _navigate_previous(self):
        if self._current_index > 0:
            self._filmstrip.setCurrentRow(self._current_index - 1)

    def _navigate_next(self):
        if self._current_index < len(self._tiles) - 1:
            self._filmstrip.setCurrentRow(self._current_index + 1)

    # ── 종료 ───────────────────────────────────────────────────

    def closeEvent(self, event):
        self._db_poll_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
