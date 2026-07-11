# PCB 결함 리뷰 스테이션 — DB에서 REVIEW/FAIL 결함을 실시간으로 수신하여 표시

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QStatusBar, QLabel, QProgressBar,
    QShortcut,
)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSlot, QTimer, QEvent
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QIcon, QKeySequence,
)

from vision_viewer import VisionViewer, confidence_color

# ── 프로젝트 루트 ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── DB 연동 ────────────────────────────────
try:
    from db.database import (
        init_db as _db_init,
        fetch_review_defects as _db_fetch,
        fetch_tile_defects as _db_fetch_tile_defects,
        get_settings as _db_get_settings,
        save_defect_verdict as _db_save_verdict,
        update_defect_bbox as _db_update_bbox,
        get_tile_image as _db_get_tile_image,
    )
    _DB_ENABLED = True
except Exception as _e:
    print(f"[DB] 연동 비활성화: {_e}")
    _DB_ENABLED = False

THUMB_SIZE = 96
BORDER_WIDTH = 3
DEBOUNCE_SEC = 0.10
CROP_PAD_RATIO = 0.3
# 한 번의 폴링(3초)에서 처리할 최대 결함 수 — REVIEW/FAIL이 폭주해 한 번에 수십~수백 건이
# 도착해도 전부 처리하면 UI가 통째로 멈추므로, 초과분은 다음 폴링으로 미뤄 분산한다
MAX_DEFECTS_PER_POLL_TICK = 3

BORDER_COLORS = {
    "pending": QColor(128, 128, 128),
    "pass":    QColor(0, 200, 0),
    "fail":    QColor(255, 50, 50),
}


# ── DefectEntry ───────────────────────────────────────────────
class DefectEntry:
    """리뷰 대기 결함 1건 — 필름스트립 엔트리 1:1."""
    def __init__(self, defect_id: int, tile_id: int,
                 class_id: int, class_name: str, confidence: float,
                 bbox_abs: list, ai_verdict: str, user_verdict: str | None = None):
        self.defect_id = defect_id
        self.tile_id = tile_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_abs = bbox_abs
        self.ai_verdict = ai_verdict
        self.user_verdict = user_verdict or "pending"

    def as_det_dict(self) -> dict:
        """VisionViewer.set_detections()가 기대하는 dict 형태로 변환."""
        return {
            "bbox_abs": self.bbox_abs,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
        }


def _compute_crop(img: np.ndarray, bbox_abs: list) -> np.ndarray:
    """결함 영역 + 패딩을 크롭하여 썸네일 크기로 리사이즈하되,
    비율을 유지하며 정사각형으로 패딩하여 왜곡을 방지합니다."""
    x1, y1, x2, y2 = bbox_abs
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = bw * CROP_PAD_RATIO
    pad_y = bh * CROP_PAD_RATIO

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)

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
    return cv2.resize(square_crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)


# ── MainWindow ────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """REVIEW/FAIL 결함 실시간 수신 + 표시 리뷰 스테이션.

    단축키:
    - Space → Pass (양품/오탐)
    - 1~6   → Fail (결함 클래스)
    - ←/→   → 이전/다음 결함 이동
    - Shift  → 오버레이 숨김 (Hold)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepPCB Review Station")
        self.setMinimumSize(1280, 800)

        self._defects: list[DefectEntry] = []
        self._tile_cache: dict[int, np.ndarray] = {}
        self._tile_defects: dict[int, list[int]] = {}  # tile_id -> [defect_id, ...]
        self._current_index: int = -1
        self._last_verdict_time: float = 0.0
        self._last_shown_id: int = 0
        self._last_session_id: str = ""

        # DB에서 읽어온 현재 설정 (폴링마다 갱신, class_name 키) — 색상 판정에만 사용
        self._per_class_bands: dict = {
            cls: (0.30, 0.70)
            for cls in ["open", "short", "mousebite", "spur", "copper", "pinhole"]
        }

        self._db_poll_timer = QTimer(self)
        self._db_poll_timer.setInterval(3000)
        self._db_poll_timer.timeout.connect(self._poll_db)

        self._init_ui()
        self._connect_signals()
        QApplication.instance().installEventFilter(self)

        if _DB_ENABLED:
            try:
                _db_init()
                s = _db_get_settings()
                self._per_class_bands = self._parse_bands(s)
                self._last_session_id = s.get("db_session_id", "")
            except Exception as e:
                print(f"[DB] 초기 설정 로드 실패: {e}")
            self._db_poll_timer.start()
            self._status_label.setText("[DB 연결됨] REVIEW/FAIL 결함 대기 중... (3초마다 갱신)")
        else:
            self._status_label.setText("DB 비활성화")

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

        # GlobalView — 현재 결함이 속한 타일 전체 미리보기
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
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda _dx=dx, _dy=dy: self._local_view.pan(_dx, _dy))

        sc_q = QShortcut(QKeySequence(Qt.Key_Q), self)
        sc_q.setContext(Qt.ApplicationShortcut)
        sc_q.activated.connect(lambda: self._local_view.zoom(zoom_in=False))
        sc_e = QShortcut(QKeySequence(Qt.Key_E), self)
        sc_e.setContext(Qt.ApplicationShortcut)
        sc_e.activated.connect(lambda: self._local_view.zoom(zoom_in=True))

        # ESC: 현재 결함 판정 취소 (pending으로 리셋)
        self._sc_cancel = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._sc_cancel.setContext(Qt.ApplicationShortcut)
        self._sc_cancel.activated.connect(self._cancel_current_verdict)

        # LocalView에서 bbox 드래그 편집(이동/리사이즈) 완료 시
        self._local_view.bbox_edited.connect(self._on_bbox_edited)

    # ── 설정 파싱 ──────────────────────────────────────────────

    def _parse_bands(self, s: dict) -> dict:
        """DB settings dict → {class_name: (review_min, review_max)} (0.0~1.0 비율).

        class_id(모델 가중치에 고정된 정수)가 아니라 이름을 키로 쓴다 — app_front에서
        클래스 목록을 자유롭게 추가/삭제하면 순서/개수가 바뀔 수 있기 때문이다.
        이 밴드는 색상 판정에만 쓰인다 — 결함의 REVIEW/FAIL 여부 자체는 DB에 저장된
        defects.verdict(app_front가 저장 시점에 판정한 값)를 그대로 신뢰한다.
        """
        bands = {}
        try:
            class_names = json.loads(s.get("defect_classes", "[]"))
        except Exception:
            class_names = []
        for name in class_names:
            r_min = int(s.get(f"review_min_{name}", 30)) / 100.0
            r_max = int(s.get(f"review_max_{name}", 70)) / 100.0
            bands[name] = (r_min, r_max)
        return bands

    # ── DB 폴링 ────────────────────────────────────────────────

    def _poll_db(self):
        """3초마다 DB 설정 동기화 + 새 REVIEW/FAIL 결함 조회 → FilmStrip 추가."""
        if not _DB_ENABLED:
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

        # per-class bands 갱신 (색상 판정용, 클래스 추가/삭제도 다음 폴링에서 자동 반영)
        self._per_class_bands = self._parse_bands(s)

        # ── 새 결함 조회 ─────────────────────────────────────────
        try:
            new_defects = _db_fetch(after_id=self._last_shown_id)
        except Exception as e:
            self._status_label.setText(f"[DB 조회 오류] {e}")
            return

        # REVIEW/FAIL 폭주 시 한 틱에 전부 처리하면 UI가 멈출 수 있으므로,
        # 이번 틱은 앞쪽 MAX_DEFECTS_PER_POLL_TICK건만 처리하고 나머지는 다음 틱(3초 후)으로 미룬다.
        to_process = new_defects[:MAX_DEFECTS_PER_POLL_TICK]
        backlog = len(new_defects) - len(to_process)
        for defect_row in to_process:
            self._last_shown_id = defect_row["id"]
            self._process_defect(defect_row)

        pending = sum(1 for d in self._defects if d.user_verdict == "pending")
        total = len(self._defects)
        status = f"[DB 연결됨] 결함 {total}건 수신 | 미검토 {pending}건"
        if backlog > 0:
            status += f" | 처리 대기 {backlog}건(다음 갱신에 이어서 처리)"
        self._status_label.setText(status)

    def _on_db_reset(self):
        """DB 초기화 감지 시 FilmStrip/상태 전체 리셋."""
        self._defects.clear()
        self._tile_cache.clear()
        self._tile_defects.clear()
        self._filmstrip.clear()
        self._last_shown_id = 0
        self._current_index = -1
        self._local_view.clear_all()
        self._global_scene.clear()
        self._status_label.setText("DB가 초기화되었습니다. 새 검사를 기다리는 중...")

    # ── 결함 처리 ──────────────────────────────────────────────

    def _ensure_tile_image(self, tile_id: int) -> np.ndarray | None:
        """타일 이미지를 캐시에서 찾거나 DB에서 조회해 디코딩 후 캐시에 저장."""
        img = self._tile_cache.get(tile_id)
        if img is not None:
            return img
        try:
            blob = _db_get_tile_image(tile_id)
        except Exception as e:
            print(f"[DB] 타일 이미지 조회 실패 (tile_id={tile_id}): {e}")
            return None
        if blob is None:
            return None
        buf = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        self._tile_cache[tile_id] = img
        return img

    def _process_defect(self, defect_row: dict):
        """DB 결함 행 → DefectEntry 생성 → FilmStrip 추가."""
        tile_img = self._ensure_tile_image(defect_row["tile_id"])
        if tile_img is None:
            return

        entry = DefectEntry(
            defect_id=defect_row["id"],
            tile_id=defect_row["tile_id"],
            class_id=defect_row["class_id"],
            class_name=defect_row["class_name"],
            confidence=defect_row["confidence"],
            bbox_abs=defect_row["bbox_abs"],
            ai_verdict=defect_row["verdict"],
            user_verdict=defect_row.get("user_verdict"),
        )
        self._defects.append(entry)
        self._tile_defects.setdefault(entry.tile_id, []).append(entry.defect_id)

        thumb_bgr = _compute_crop(tile_img, entry.bbox_abs)
        color = BORDER_COLORS.get(
            entry.user_verdict if entry.user_verdict in BORDER_COLORS else "fail",
            BORDER_COLORS["pending"]
        )
        pixmap = self._make_thumbnail(thumb_bgr, color, entry.user_verdict)

        item = QListWidgetItem()
        item.setIcon(QIcon(pixmap))
        item.setSizeHint(QSize(THUMB_SIZE + 8, THUMB_SIZE + 8))
        item.setToolTip(
            f"[{entry.ai_verdict}] Defect #{entry.defect_id}\n"
            f"{entry.class_name}  신뢰도: {entry.confidence:.2f}"
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
        if not (0 <= index < len(self._defects)):
            return
        entry = self._defects[index]
        color = BORDER_COLORS.get(
            entry.user_verdict if entry.user_verdict in BORDER_COLORS else "fail",
            BORDER_COLORS["pending"]
        )
        tile_img = self._tile_cache.get(entry.tile_id)
        if tile_img is None:
            return
        thumb = _compute_crop(tile_img, entry.bbox_abs)
        pixmap = self._make_thumbnail(thumb, color, entry.user_verdict)
        item = self._filmstrip.item(index)
        if item:
            item.setIcon(QIcon(pixmap))

    # ── 뷰 갱신 ───────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_filmstrip_selection(self, index: int):
        if not (0 <= index < len(self._defects)):
            return
        self._current_index = index
        entry = self._defects[index]
        tile_img = self._tile_cache.get(entry.tile_id)
        if tile_img is None:
            return

        # GlobalView — 이 결함이 속한 타일 전체 미리보기 (형제 엔트리를 클릭해도 동일)
        self._global_scene.clear()
        rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self._global_scene.addPixmap(pixmap)
        self._global_scene.setSceneRect(QRectF(pixmap.rect()))
        self._global_view.fitInView(
            self._global_scene.sceneRect(), Qt.KeepAspectRatio
        )

        # LocalView — 같은 타일의 형제 결함도 함께 그리되(참고용) 이 결함만 강조,
        # 줌 대상은 이 결함 1건 (형제마다 위치가 다르므로 union이 아니라 자신 기준).
        # 필름스트립은 REVIEW만 갖고 있으므로, FAIL 형제까지 보여주려면 DB에서
        # 이 타일의 REVIEW+FAIL 전체를 직접 조회해야 한다(self._defects로는 부족).
        try:
            sibling_dets = _db_fetch_tile_defects(entry.tile_id)
        except Exception as e:
            print(f"[DB] 형제 결함 조회 실패 (tile_id={entry.tile_id}): {e}")
            sibling_dets = [{"id": entry.defect_id, **entry.as_det_dict()}]
        highlight_index = next(
            (i for i, d in enumerate(sibling_dets) if d["id"] == entry.defect_id), -1
        )

        self._local_view.set_image(tile_img)
        self._local_view.set_detections(
            sibling_dets,
            highlight_index=highlight_index,
            per_class_bands=self._per_class_bands,
        )
        self._local_view.zoom_to_detections([entry.as_det_dict()], pad_ratio=1.0)

        self._update_status()

    def _on_bbox_edited(self, new_bbox_abs: list):
        """LocalView에서 드래그 편집(이동/리사이즈)이 끝났을 때 호출 — DB에 1회 저장."""
        if not (0 <= self._current_index < len(self._defects)):
            return
        entry = self._defects[self._current_index]
        entry.bbox_abs = new_bbox_abs
        if _DB_ENABLED:
            try:
                _db_update_bbox(entry.defect_id, new_bbox_abs)
            except Exception as e:
                print(f"[DB] bbox 저장 실패 (defect_id={entry.defect_id}): {e}")
        self._update_thumbnail(self._current_index)

    def _update_status(self):
        reviewed = sum(1 for d in self._defects if d.user_verdict != "pending")
        total = len(self._defects)
        passed = sum(1 for d in self._defects if d.user_verdict == "pass")
        failed = sum(1 for d in self._defects if d.user_verdict not in ("pending", "pass"))
        cur = self._current_index + 1 if self._current_index >= 0 else 0
        self._status_label.setText(
            f"[{cur}/{total}]  Pass: {passed} | Fail: {failed} | 미검토: {total - reviewed}"
        )

    # ── 단축키 ─────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control and not event.isAutoRepeat():
            self._focus_first_pending()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        """Shift 오버레이 숨김을 포커스와 무관하게 애플리케이션 전역에서 감지.

        MainWindow.keyPressEvent/keyReleaseEvent는 MainWindow가 포커스를 가질 때만
        호출된다. GlobalView(QGraphicsView) 등 자식 위젯이 포커스를 가져가면 Shift
        press/release가 비대칭으로 씹혀 오버레이가 숨겨진 채 고착될 수 있으므로,
        애플리케이션 전역 이벤트 필터로 감지해 포커스 위치와 무관하게 동작시킨다.
        """
        etype = event.type()
        if etype == QEvent.KeyPress and event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(False)
        elif etype == QEvent.KeyRelease and event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self._local_view.set_overlay_visible(True)
        elif etype == QEvent.ApplicationDeactivate:
            self._local_view.set_overlay_visible(True)
        return super().eventFilter(obj, event)

    def _verdict_current(self, verdict: str):
        now = time.time()
        if now - self._last_verdict_time < DEBOUNCE_SEC:
            return
        self._last_verdict_time = now
        if not (0 <= self._current_index < len(self._defects)):
            return

        entry = self._defects[self._current_index]
        was_pending = entry.user_verdict == "pending"
        entry.user_verdict = verdict

        # DB에 사용자 판정 저장 (이 결함 1건만 — 같은 타일의 다른 결함은 독립 판정)
        if _DB_ENABLED:
            try:
                _db_save_verdict(entry.defect_id, verdict)
            except Exception as e:
                print(f"[DB] 판정 저장 실패 (defect_id={entry.defect_id}): {e}")

        self._update_thumbnail(self._current_index)
        if was_pending:
            self._advance_to_next()

    def _advance_to_next(self):
        for i in range(self._current_index + 1, len(self._defects)):
            if self._defects[i].user_verdict == "pending":
                self._filmstrip.setCurrentRow(i)
                return

    def _navigate_previous(self):
        if self._current_index > 0:
            self._filmstrip.setCurrentRow(self._current_index - 1)

    def _navigate_next(self):
        if self._current_index < len(self._defects) - 1:
            self._filmstrip.setCurrentRow(self._current_index + 1)

    # ── ESC / Ctrl ─────────────────────────────────────────────

    def _remove_entry_at(self, index: int):
        """결함 엔트리를 필름스트립/목록에서 제거.

        Qt는 현재 선택된 행을 takeItem()으로 지우면 내부 currentRow를
        자동으로 재조정하는데, 그 값이 이후 우리가 호출하는 setCurrentRow()의
        인자와 우연히 같아지면 "값이 안 바뀌었다"고 보고 currentRowChanged
        시그널을 발동시키지 않는다. 화면 갱신(_on_filmstrip_selection)이
        오직 이 시그널에만 걸려 있으면 갱신이 누락될 수 있으므로, 시그널
        발동 여부와 무관하게 화면 갱신 함수를 항상 직접 호출한다.
        """
        if not (0 <= index < len(self._defects)):
            return

        entry = self._defects[index]
        siblings = self._tile_defects.get(entry.tile_id)
        if siblings and entry.defect_id in siblings:
            siblings.remove(entry.defect_id)
            if not siblings:
                del self._tile_defects[entry.tile_id]
                self._tile_cache.pop(entry.tile_id, None)

        self._filmstrip.currentRowChanged.disconnect(self._on_filmstrip_selection)
        try:
            del self._defects[index]
            self._filmstrip.takeItem(index)
            new_index = min(index, len(self._defects) - 1) if self._defects else -1
            if new_index >= 0:
                self._filmstrip.setCurrentRow(new_index)
        finally:
            self._filmstrip.currentRowChanged.connect(self._on_filmstrip_selection)

        if new_index == -1:
            self._current_index = -1
            self._local_view.clear_all()
            self._global_scene.clear()
            self._update_status()
        else:
            self._on_filmstrip_selection(new_index)

    def _focus_first_pending(self):
        """Ctrl: 아직 판정하지 않은 가장 앞쪽 결함으로 포커스 이동."""
        for i, entry in enumerate(self._defects):
            if entry.user_verdict == "pending":
                if i != self._current_index:
                    self._filmstrip.setCurrentRow(i)
                return

    def _cancel_current_verdict(self):
        """ESC: 현재 결함의 판정(1~6 또는 PASS)을 취소하고 pending으로 리셋."""
        if not (0 <= self._current_index < len(self._defects)):
            return

        entry = self._defects[self._current_index]
        if entry.user_verdict == "pending":
            return

        entry.user_verdict = "pending"
        if _DB_ENABLED:
            try:
                _db_save_verdict(entry.defect_id, None)
            except Exception as e:
                print(f"[DB] 판정 취소 실패 (defect_id={entry.defect_id}): {e}")

        self._update_thumbnail(self._current_index)

    # ── 종료 ───────────────────────────────────────────────────

    def closeEvent(self, event):
        self._db_poll_timer.stop()
        super().closeEvent(event)
