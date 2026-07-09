# PCB 실시간 검사 모니터링 메인 윈도우 (자동 AOI 시뮬레이션)

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem,
    QStatusBar, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QApplication, QAction, QShortcut, QDialog, QFormLayout,
    QDialogButtonBox, QSpinBox, QCheckBox, QGroupBox,
    QGridLayout, QFrame, QSizePolicy, QPushButton,
)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSlot, QTimer, QEvent
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QIcon, QKeySequence,
    QFont,
)

from vision_viewer import VisionViewer
from global_view import GlobalView
from inspection_worker import InspectionWorker
from class_band_table import ClassBandTableWidget

# ── 프로젝트 루트 ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ── DB 연동 (실패해도 앱 동작은 유지) ──────────
import sys as _sys
_sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from db.database import (
        init_db as _db_init,
        upsert_tile as _db_upsert,
        clear_all as _db_clear,
        get_settings as _db_get_settings,
        update_settings as _db_update_settings,
        get_db_stats as _db_get_stats,
        force_reset_settings_to_defaults as _db_force_reset_settings,
        load_default_settings_raw as _db_load_defaults_raw,
        save_default_settings_raw as _db_save_defaults_raw,
    )
    _DB_ENABLED = True
except Exception as _e:
    print(f"[DB] 연동 비활성화: {_e}")
    _DB_ENABLED = False

# ── 상수 ───────────────────────────────────
THUMB_SIZE = 96
BORDER_WIDTH = 3
FILMSTRIP_MAX_ITEMS = 200  # 24시간 상시 구동 대비 최근 N개만 유지

VERDICT_COLORS = {
    "PASS":   QColor(0, 200, 0),
    "FAIL":   QColor(255, 50, 50),
    "REVIEW": QColor(255, 200, 0),
}


# ── 공통 스타일 ────────────────────────────────────────────────

_DIALOG_STYLE = """
    QDialog { background-color: #1e1e1e; color: #ccc; }
    QLabel { color: #ccc; }
    QTableWidget { background-color: #1a1a1a; color: #ccc;
                    gridline-color: #333; border: 1px solid #333; }
    QTableWidget::item { padding: 2px; }
    QTableWidget::item:selected { background-color: #2a4a7a; }
    QHeaderView::section { background-color: #2a2a2a; color: #ccc;
                             border: 1px solid #333; padding: 4px; }
    QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; color: #ccc; padding: 2px; }
    QCheckBox { color: #ccc; }
    QPushButton { background-color: #333; color: #ccc; padding: 5px; }
    QPushButton:hover { background-color: #444; }
"""


def _settings_to_classes_list(settings: dict) -> list:
    """settings dict(defect_classes JSON + review_min_*/review_max_*)를
    [{"name","review_min","review_max"}, ...] 형태로 변환."""
    names = json.loads(settings.get("defect_classes", "[]"))
    return [
        {
            "name": name,
            "review_min": int(settings.get(f"review_min_{name}", 30)),
            "review_max": int(settings.get(f"review_max_{name}", 70)),
        }
        for name in names
    ]


# ── SettingsDialog ────────────────────────────────────────────

class SettingsDialog(QDialog):
    """검사 파라미터 설정 다이얼로그 (클래스별 REVIEW 밴드, 타일 크기/오버랩).

    이 다이얼로그에서 바꾼 값은 현재 세션의 DB에만 반영된다(공장 기본값은 안 바뀜).
    """

    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Settings")
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        # ── per-class REVIEW 밴드 테이블 (자유 추가/삭제) ──────────
        band_label = QLabel("클래스별 REVIEW 신뢰도 밴드 (하한 ≤ REVIEW < 상한, 상한 초과 → FAIL)")
        band_label.setStyleSheet("color: #aaa; font-size: 8pt;")
        layout.addWidget(band_label)

        self._class_table = ClassBandTableWidget(_settings_to_classes_list(current_settings))
        layout.addWidget(self._class_table)

        # ── 타일 크기 / 오버랩 / 경고음 ────────────────────────────
        form_layout = QFormLayout()

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(8, 1280)
        self.tile_size_spin.setSuffix(" px")
        self.tile_size_spin.setValue(int(current_settings.get("tile_size", 640)))
        self.tile_size_spin.setToolTip("정사각형 검사 단위 한 변의 길이 (8~1280px)")
        form_layout.addRow("타일 크기:", self.tile_size_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(int(current_settings.get("overlap_pct", 0)))
        self.overlap_spin.setToolTip("인접 타일 간 겹침 비율 (0~90%)")
        form_layout.addRow("오버랩:", self.overlap_spin)

        self.alert_check = QCheckBox("FAIL 검출 시 경고음")
        alert_val = current_settings.get("alert_sound", True)
        if isinstance(alert_val, str):
            alert_val = alert_val.lower() == "true"
        self.alert_check.setChecked(bool(alert_val))
        form_layout.addRow("", self.alert_check)

        layout.addLayout(form_layout)

        # ── 기본값 수정 진입점 ─────────────────────────────────────
        defaults_btn = QPushButton("기본값 수정...")
        defaults_btn.setToolTip("프로그램을 다시 시작했을 때 적용되는 공장 기본값을 수정합니다.")
        defaults_btn.clicked.connect(self._open_defaults_edit_dialog)
        layout.addWidget(defaults_btn)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setStyleSheet(_DIALOG_STYLE)

    def _open_defaults_edit_dialog(self):
        DefaultsEditDialog(self).exec_()

    def _validate_and_accept(self):
        """각 클래스에 대해 review_min < review_max 유효성 검증."""
        err = self._class_table.validate()
        if err:
            QMessageBox.warning(self, "설정 오류", err)
            return
        self.accept()

    def get_settings(self) -> dict:
        s: dict = {}
        bands = self._class_table.get_bands()
        s["defect_classes"] = json.dumps(list(bands.keys()))
        for name, (mn, mx) in bands.items():
            s[f"review_min_{name}"] = mn
            s[f"review_max_{name}"] = mx
        s["tile_size"] = self.tile_size_spin.value()
        s["overlap_pct"] = self.overlap_spin.value()
        s["alert_sound"] = self.alert_check.isChecked()
        return s


# ── DefaultsEditDialog ────────────────────────────────────────

class DefaultsEditDialog(QDialog):
    """공장 기본값(default_settings.json) 자체를 수정하는 다이얼로그.

    여기서 바꾼 값은 현재 세션에는 영향을 주지 않고, 다음 실행부터 적용된다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("기본값 수정")
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._raw = _db_load_defaults_raw() if _DB_ENABLED else {
            "defect_classes": [], "tile_size": 640, "overlap_pct": 0, "alert_sound": True,
        }

        layout = QVBoxLayout(self)

        notice = QLabel("여기서 수정한 값은 프로그램을 다시 시작했을 때부터 적용됩니다.")
        notice.setStyleSheet("color: #ffc800; font-size: 8pt;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        band_label = QLabel("공장 기본 클래스 목록 및 REVIEW 밴드")
        band_label.setStyleSheet("color: #aaa; font-size: 8pt;")
        layout.addWidget(band_label)

        self._class_table = ClassBandTableWidget(self._raw.get("defect_classes", []))
        layout.addWidget(self._class_table)

        form_layout = QFormLayout()

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(8, 1280)
        self.tile_size_spin.setSuffix(" px")
        self.tile_size_spin.setValue(int(self._raw.get("tile_size", 640)))
        form_layout.addRow("타일 크기:", self.tile_size_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(int(self._raw.get("overlap_pct", 0)))
        form_layout.addRow("오버랩:", self.overlap_spin)

        self.alert_check = QCheckBox("FAIL 검출 시 경고음")
        self.alert_check.setChecked(bool(self._raw.get("alert_sound", True)))
        form_layout.addRow("", self.alert_check)

        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._validate_and_save)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setStyleSheet(_DIALOG_STYLE)

    def _validate_and_save(self):
        err = self._class_table.validate()
        if err:
            QMessageBox.warning(self, "설정 오류", err)
            return

        reply = QMessageBox.question(
            self, "기본값 수정",
            "기본값을 수정하면 프로그램을 다시 시작해야 적용됩니다.\n계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        bands = self._class_table.get_bands()
        new_raw = {
            "defect_classes": [
                {"name": name, "review_min": mn, "review_max": mx}
                for name, (mn, mx) in bands.items()
            ],
            "tile_size": self.tile_size_spin.value(),
            "overlap_pct": self.overlap_spin.value(),
            "alert_sound": self.alert_check.isChecked(),
        }
        if _DB_ENABLED:
            try:
                _db_save_defaults_raw(new_raw)
            except Exception as e:
                QMessageBox.warning(self, "오류", f"기본값 저장 실패: {e}")
                return
        self.accept()


# ── MainWindow ────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """PCB 실시간 검사 모니터링 메인 윈도우.

    레이아웃:
    ┌────────────────────────────────────────────┐
    │  GlobalView(30%)     │ LocalView(70%)       │  ← 상단
    │  [컬러 PCB + 그리드] │ [이진화 타일 + 결함]  │
    │                      │                      │
    │  Stats Panel         │                      │
    ├────────────────────────────────────────────┤
    │       FilmStrip (타일 히스토리, 가로 스크롤)  │  ← 하단
    └────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepPCB Real-Time Inspection Monitor")
        self.setMinimumSize(1280, 800)

        self._worker: InspectionWorker | None = None
        self._paused: bool = False
        self._current_folder: str | None = None
        self._current_image_index: int = -1
        self._colored_images: dict[int, np.ndarray] = {}
        self._inspection_start_time: float = 0.0
        self._tile_count: int = 0

        # 통계 카운터
        self._stats = {
            "total_tiles": 0,
            "inspected": 0,
            "pass": 0,
            "fail": 0,
            "review": 0,
        }
        self._inference_times: list[float] = []

        # Options 설정 — app_front 구동 시 항상 파일 기본값으로 DB를 강제 초기화한 뒤 로드
        # (세션 중 Option 변경은 DB에만 반영되고 파일 기본값은 바뀌지 않는다)
        if _DB_ENABLED:
            try:
                _db_init()
                _db_force_reset_settings()
            except Exception as e:
                print(f"[DB] 기본값 강제 초기화 실패: {e}")
        self._app_settings = self._load_settings()
        self._defect_distribution = {name: 0 for name in self._current_class_names()}

        self._init_ui()
        self._init_menu()
        self._connect_signals()

        self._status_label.setText("Ready. File > Open Folder... to start.")

    # ── 설정 로드/저장 (DB 기반) ──────────────────────────────

    _FALLBACK_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]

    def _load_settings(self) -> dict:
        """DB settings 테이블에서 설정을 읽어 반환. DB가 없으면 기본값 사용."""
        defaults: dict = {"defect_classes": json.dumps(self._FALLBACK_CLASSES)}
        for cls in self._FALLBACK_CLASSES:
            defaults[f"review_min_{cls}"] = 30
            defaults[f"review_max_{cls}"] = 70
        defaults["tile_size"] = 640
        defaults["overlap_pct"] = 0
        defaults["alert_sound"] = True

        if not _DB_ENABLED:
            return defaults
        try:
            _db_init()
            raw = _db_get_settings()
            class_names = json.loads(raw.get("defect_classes", "[]")) or self._FALLBACK_CLASSES
            result: dict = {"defect_classes": json.dumps(class_names)}
            for cls in class_names:
                result[f"review_min_{cls}"] = int(raw.get(f"review_min_{cls}", 30))
                result[f"review_max_{cls}"] = int(raw.get(f"review_max_{cls}", 70))
            result["tile_size"] = int(raw.get("tile_size", 640))
            result["overlap_pct"] = int(raw.get("overlap_pct", 0))
            alert_val = raw.get("alert_sound", "true")
            result["alert_sound"] = alert_val if isinstance(alert_val, bool) else alert_val.lower() == "true"
            return result
        except Exception as e:
            print(f"[DB] 설정 로드 실패, 기본값 사용: {e}")
            return defaults

    def _current_class_names(self) -> list:
        """현재 _app_settings에서 활성 클래스 이름 리스트를 파싱."""
        try:
            return json.loads(self._app_settings.get("defect_classes", "[]"))
        except Exception:
            return list(self._FALLBACK_CLASSES)

    def _save_settings(self, settings: dict) -> None:
        """설정을 DB settings 테이블에 저장."""
        if not _DB_ENABLED:
            return
        try:
            _db_update_settings(settings)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"설정 저장 실패: {e}")

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

        # 좌측: GlobalView + Stats
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._global_view = GlobalView()
        self._global_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_layout.addWidget(self._global_view, stretch=6)

        # 통계 패널
        stats_group = QGroupBox("Inspection Statistics")
        stats_group.setMaximumHeight(220)
        stats_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a1a; color: #ccc;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 6px; padding-top: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px;
            }
            QLabel { color: #ccc; font-size: 9pt; }
        """)
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(2)

        def _stat_label(text, color="#ccc"):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-size: 9pt;")
            return lbl

        # 행 0: 진행률
        stats_layout.addWidget(_stat_label("Inspected:"), 0, 0)
        self._lbl_inspected = _stat_label("0 / 0")
        stats_layout.addWidget(self._lbl_inspected, 0, 1)

        # 행 1-3: PASS / FAIL / REVIEW
        stats_layout.addWidget(_stat_label("PASS:", "#00c800"), 1, 0)
        self._lbl_pass = _stat_label("0 (0.0%)", "#00c800")
        stats_layout.addWidget(self._lbl_pass, 1, 1)

        stats_layout.addWidget(_stat_label("FAIL:", "#ff3232"), 2, 0)
        self._lbl_fail = _stat_label("0 (0.0%)", "#ff3232")
        stats_layout.addWidget(self._lbl_fail, 2, 1)

        stats_layout.addWidget(_stat_label("REVIEW:", "#ffc800"), 3, 0)
        self._lbl_review = _stat_label("0 (0.0%)", "#ffc800")
        stats_layout.addWidget(self._lbl_review, 3, 1)

        # 행 4: FPY
        stats_layout.addWidget(_stat_label("FPY:"), 4, 0)
        self._lbl_fpy = _stat_label("0.0%")
        stats_layout.addWidget(self._lbl_fpy, 4, 1)

        # 행 5: Throughput
        stats_layout.addWidget(_stat_label("Throughput:"), 5, 0)
        self._lbl_throughput = _stat_label("0.0 tiles/min")
        stats_layout.addWidget(self._lbl_throughput, 5, 1)

        # 행 6: Elapsed
        stats_layout.addWidget(_stat_label("Elapsed:"), 6, 0)
        self._lbl_elapsed = _stat_label("00:00:00")
        stats_layout.addWidget(self._lbl_elapsed, 6, 1)

        # 행 7: 결함 분포 헤더
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        stats_layout.addWidget(sep, 7, 0, 1, 2)

        stats_layout.addWidget(_stat_label("Defect Distribution:"), 8, 0, 1, 2)
        self._lbl_defect_dist = _stat_label("-")
        self._lbl_defect_dist.setWordWrap(True)
        stats_layout.addWidget(self._lbl_defect_dist, 9, 0, 1, 2)

        left_layout.addWidget(stats_group, stretch=4)
        self._h_splitter.addWidget(left_widget)

        # 우측: LocalView
        self._local_view = VisionViewer()
        self._local_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._h_splitter.addWidget(self._local_view)

        self._h_splitter.setStretchFactor(0, 3)
        self._h_splitter.setStretchFactor(1, 7)
        self._h_splitter.setCollapsible(0, False)
        self._h_splitter.setCollapsible(1, False)
        self._h_splitter.setSizes([300, 700])

        # ── 하단: FilmStrip ──
        # app_front는 24시간 상시 구동되며 썸네일 클릭 선택/개별 판정 의미가 없으므로
        # 클릭 선택 하이라이트와 마우스 휠 스크롤 이동을 비활성화한다(app_back과 다른 정책).
        self._filmstrip = QListWidget()
        self._filmstrip.setFocusPolicy(Qt.NoFocus)
        self._filmstrip.setSelectionMode(QListWidget.NoSelection)
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
        """)
        self._filmstrip.viewport().installEventFilter(self)
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

        # 일시정지 표시 라벨
        self._pause_label = QLabel("")
        self._pause_label.setStyleSheet("color: #ff9900; font-weight: bold;")
        self._status_bar.addPermanentWidget(self._pause_label)

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

        # 경과 시간 타이머
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def _init_menu(self):
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
            "QMenuBar { background: #1a1a1a; color: #ccc; }"
            "QMenuBar::item:selected { background: #333; }"
            "QMenu { background: #2a2a2a; color: #ccc; }"
            "QMenu::item:selected { background: #3a7bd5; }"
        )

        # File 메뉴
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._select_folder_and_start)
        file_menu.addAction(open_action)

        # Database 메뉴 (File과 Option 사이)
        db_menu = menu_bar.addMenu("Database")
        stats_action = QAction("DB 통계...", self)
        stats_action.triggered.connect(self._show_db_stats)
        db_menu.addAction(stats_action)
        reset_action = QAction("DB 초기화...", self)
        reset_action.triggered.connect(self._reset_db)
        db_menu.addAction(reset_action)

        # Option 메뉴
        option_menu = menu_bar.addMenu("Option")
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        option_menu.addAction(settings_action)

    def _open_settings_dialog(self):
        dialog = SettingsDialog(self._app_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            self._app_settings = new_settings
            self._save_settings(self._app_settings)
            # 진행 중인 Worker는 중단하지 않음. 다음 폴더 오픈 시 새 설정 적용.

    def _show_db_stats(self):
        """DB 통계 다이얼로그 표시."""
        if not _DB_ENABLED:
            QMessageBox.warning(self, "DB 비활성화", "DB 연동이 비활성화되어 있습니다.")
            return
        try:
            stats = _db_get_stats()
            total = stats.get("_total", 0)
            db_bytes = stats.get("_db_bytes", 0)
            if db_bytes >= 1024 ** 2:
                size_str = f"{db_bytes / 1024 ** 2:.1f} MB"
            elif db_bytes >= 1024:
                size_str = f"{db_bytes / 1024:.1f} KB"
            else:
                size_str = f"{db_bytes} B"
            msg = (
                f"총 타일 수: {total}\n"
                f"  PASS : {stats.get('PASS', 0)}\n"
                f"  FAIL : {stats.get('FAIL', 0)}\n"
                f"  REVIEW: {stats.get('REVIEW', 0)}\n\n"
                f"DB 파일 크기: {size_str}"
            )
            QMessageBox.information(self, "DB 통계", msg)
        except Exception as e:
            QMessageBox.warning(self, "오류", f"DB 통계 조회 실패: {e}")

    def _reset_db(self):
        """DB 초기화 (tiles 삭제). 진행 중인 Worker는 중단하지 않음.
        UI 통계 패널(숫자)은 DB와 맞춰 리셋하되, 필름스트립/GlobalView/LocalView 화면은
        진행 중인 검사가 갑자기 비어 보이지 않도록 그대로 유지한다.
        """
        reply = QMessageBox.question(
            self, "DB 초기화",
            "모든 검사 데이터를 삭제하시겠습니까?\n"
            "진행 중인 검사는 중단되지 않습니다.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if not _DB_ENABLED:
            return
        try:
            _db_clear()
            self._stats = {"total_tiles": 0, "inspected": 0, "pass": 0, "fail": 0, "review": 0}
            self._defect_distribution = {name: 0 for name in self._current_class_names()}
            self._update_statistics_display()
            self._status_label.setText("DB가 초기화되었습니다.")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"DB 초기화 실패: {e}")

    def _connect_signals(self):
        # Space 키: 일시정지/재개 (마우스 포커스 무관)
        self._shortcut_pause = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._shortcut_pause.setAutoRepeat(False)
        self._shortcut_pause.setContext(Qt.ApplicationShortcut)
        self._shortcut_pause.activated.connect(self._toggle_pause)

    # ── 검사 시작 ──────────────────────────────────────────────

    def _select_folder_and_start(self):
        default_dir = str(_PROJECT_ROOT / "merged_data")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder for Inspection", default_dir
        )
        if not folder:
            self._status_label.setText("No folder selected. File > Open to start.")
            return

        if self._current_folder is not None:
            reply = QMessageBox.question(
                self, "Open New Folder",
                "새 폴더를 열면 진행 중인 검사와 모든 검사 데이터(DB 포함)가 삭제되고\n"
                "처음부터 다시 시작합니다. 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._start_inspection(folder)

    def _start_inspection(self, folder):
        self._current_folder = folder

        # 기존 워커 정리
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.blockSignals(True)
            self._worker.wait(2000)
            self._worker = None

        # 상태 초기화
        self._paused = False
        self._pause_label.setText("")
        self._current_image_index = -1
        self._last_grid_image_index = -1
        self._colored_images.clear()
        self._tile_count = 0
        self._stats = {"total_tiles": 0, "inspected": 0, "pass": 0, "fail": 0, "review": 0}
        self._defect_distribution = {name: 0 for name in self._current_class_names()}
        self._inference_times.clear()
        self._filmstrip.clear()
        self._global_view.clear_all()
        self._local_view.clear_all()

        # 이진화 이미지만 필터 (_colored.png 제외)
        folder_path = Path(folder)
        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and "_colored" not in p.stem
        )
        if not image_paths:
            QMessageBox.warning(self, "Warning", f"No images found in:\n{folder}")
            return

        # 가중치 파일 경로
        weight_paths = []
        for i in range(1, 6):
            wp = _PROJECT_ROOT / "weights" / f"best_fold_{i}.pt"
            weight_paths.append(str(wp))

        # 설정값: per-class REVIEW 밴드 (이름 키, 0.0~1.0 비율로 변환)
        per_class_bands = {
            name: (
                self._app_settings.get(f"review_min_{name}", 30) / 100.0,
                self._app_settings.get(f"review_max_{name}", 70) / 100.0,
            )
            for name in self._current_class_names()
        }
        tile_size = int(self._app_settings.get("tile_size", 640))
        overlap_pct = int(self._app_settings.get("overlap_pct", 0))

        # GPU/CPU 자동 판별
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        self._worker = InspectionWorker(
            weight_paths=weight_paths,
            per_class_bands=per_class_bands,
            tile_size=tile_size,
            overlap_pct=overlap_pct,
            device=device,
        )
        self._worker.set_image_paths(image_paths)
        self._worker.models_loaded.connect(self._on_models_loaded)
        self._worker.image_started.connect(self._on_image_started)
        self._worker.tile_inspected.connect(self._on_tile_inspected)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)

        # DB 초기화 — 두 번째 폴더부터는 위 통합 확인창에서 이미 동의를 받았으므로
        # 조건 없이 항상 초기화한다(최초 실행 시에는 지울 데이터가 없어도 안전하게 init만 수행).
        if _DB_ENABLED:
            try:
                _db_init()
                _db_clear()
            except Exception as e:
                print(f"[DB] 초기화 실패: {e}")

        self._status_label.setText("Loading 5 K-Fold models...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate

        self._worker.start()

    # ── Worker Signal Slots ────────────────────────────────────

    @pyqtSlot()
    def _on_models_loaded(self):
        self._status_label.setText("Models loaded. Inspection running...")
        self._inspection_start_time = time.time()
        self._elapsed_timer.start()

    @pyqtSlot(int, str)
    def _on_image_started(self, img_index, filename):
        """새 이미지 검사 시작. 컬러 이미지를 GlobalView에 로드."""
        self._current_image_index = img_index

        # 컬러 이미지 로드 (캐시)
        if img_index not in self._colored_images:
            colored_path = self._find_colored_image(filename)
            if colored_path:
                colored_img = cv2.imread(str(colored_path))
                if colored_img is not None:
                    self._colored_images[img_index] = colored_img

        # GlobalView에 컬러 이미지 설정
        if img_index in self._colored_images:
            self._global_view.set_image(self._colored_images[img_index])

        self._status_label.setText(
            f"Inspecting image {img_index + 1}: {filename}"
        )

    @pyqtSlot(dict)
    def _on_tile_inspected(self, result):
        """타일 1개 검사 완료. UI 전체 갱신."""
        img_index = result["image_index"]
        row = result["grid_row"]
        col = result["grid_col"]
        rows = result["grid_rows"]
        cols = result["grid_cols"]
        verdict = result["verdict"]
        detections = result["detections"]
        tile_bgr = result["tile_bgr"]
        thumb_bgr = result["thumb_bgr"]
        inference_ms = result["inference_time_ms"]

        # 그리드 초기화 (새 이미지의 첫 타일 도착 시)
        if not hasattr(self, "_last_grid_image_index") or self._last_grid_image_index != img_index:
            self._last_grid_image_index = img_index
            tile_size = self._worker.tile_size if self._worker else 640
            self._global_view.set_grid(rows, cols, cell_w=tile_size, cell_h=tile_size)

        # GlobalView 업데이트
        self._global_view.set_camera_position(row, col)
        self._global_view.update_cell(row, col, verdict)

        # LocalView 업데이트 (이진화 타일 표시)
        self._local_view.set_image(tile_bgr)
        if detections:
            self._local_view.set_detections(detections)

        # FilmStrip 썸네일 추가
        self._add_thumbnail(thumb_bgr, verdict, result)

        # 통계 업데이트
        self._stats["inspected"] += 1
        if verdict == "PASS":
            self._stats["pass"] += 1
        elif verdict == "FAIL":
            self._stats["fail"] += 1
        elif verdict == "REVIEW":
            self._stats["review"] += 1

        self._inference_times.append(inference_ms)
        self._tile_count += 1

        # 결함 분포 업데이트
        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name in self._defect_distribution:
                self._defect_distribution[cls_name] += 1

        self._update_statistics_display()

        # FAIL 경고음
        alert_val = self._app_settings.get("alert_sound", True)
        if isinstance(alert_val, str):
            alert_val = alert_val.lower() == "true"
        if verdict == "FAIL" and alert_val:
            self._play_alert_sound()

        # DB 기록 — 타일 이미지(PNG BLOB) + 판정 결과 (같은 위치는 최신으로 교체)
        if _DB_ENABLED:
            try:
                _db_upsert(tile_bgr, verdict, result["image_path"], row, col)
            except Exception as e:
                print(f"[DB] 타일 기록 실패 (row={row},col={col}): {e}")

    @pyqtSlot(int, int)
    def _on_progress(self, current, total):
        self._stats["total_tiles"] = total
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)
        self._progress_bar.setVisible(True)

    @pyqtSlot()
    def _on_all_done(self):
        self._progress_bar.setVisible(False)
        self._elapsed_timer.stop()
        self._global_view.clear_camera()
        self._update_statistics_display()

        self._status_label.setText(
            f"Inspection complete. {self._stats['inspected']} tiles inspected."
        )

    @pyqtSlot(str)
    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._status_label.setText(f"Error: {msg}")

    # ── 일시정지/재개 ──────────────────────────────────────────

    def _toggle_pause(self):
        if not self._worker or not self._worker.isRunning():
            return

        if self._paused:
            self._worker.resume()
            self._paused = False
            self._pause_label.setText("")
            self._status_label.setText("Inspection resumed.")
            self._elapsed_timer.start()
        else:
            self._worker.pause()
            self._paused = True
            self._pause_label.setText("⏸ PAUSED")
            self._status_label.setText("Inspection paused. Press Space to resume.")
            self._elapsed_timer.stop()

    # ── 썸네일 생성 ────────────────────────────────────────────

    def _add_thumbnail(self, thumb_bgr, verdict, result):
        """타일 썸네일을 FilmStrip에 추가."""
        color = VERDICT_COLORS.get(verdict, QColor(128, 128, 128))
        pixmap = self._create_thumbnail_pixmap(thumb_bgr, color, verdict)

        item = QListWidgetItem()
        item.setIcon(QIcon(pixmap))
        item.setSizeHint(QSize(THUMB_SIZE + 8, THUMB_SIZE + 8))
        item.setToolTip(
            f"[{verdict}] Row {result['grid_row']}, Col {result['grid_col']}\n"
            f"Conf: {result['max_confidence']:.2f} | "
            f"{result['inference_time_ms']:.0f}ms\n"
            f"{result['image_file']}"
        )
        self._filmstrip.addItem(item)

        # 화면 밖으로 밀려난 오래된 썸네일 제거 (장기간 구동 시 메모리 누적 방지)
        while self._filmstrip.count() > FILMSTRIP_MAX_ITEMS:
            self._filmstrip.takeItem(0)

        # 자동 스크롤: 최신 타일이 보이도록
        self._filmstrip.scrollToItem(item)

    def _create_thumbnail_pixmap(self, thumb_bgr, border_color, verdict):
        """BGR numpy → 테두리 + 반투명 오버레이 QPixmap."""
        rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pixmap)

        # 반투명 색상 오버레이
        overlay = QColor(border_color)
        overlay.setAlpha(64)
        painter.fillRect(0, 0, pixmap.width(), pixmap.height(), overlay)

        # 테두리
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

    # ── 통계 표시 갱신 ──────────────────────────────────────────

    def _update_statistics_display(self):
        inspected = self._stats["inspected"]
        total = self._stats["total_tiles"]
        pass_n = self._stats["pass"]
        fail_n = self._stats["fail"]
        review_n = self._stats["review"]

        self._lbl_inspected.setText(f"{inspected} / {total}")

        if inspected > 0:
            self._lbl_pass.setText(f"{pass_n} ({pass_n/inspected*100:.1f}%)")
            self._lbl_fail.setText(f"{fail_n} ({fail_n/inspected*100:.1f}%)")
            self._lbl_review.setText(f"{review_n} ({review_n/inspected*100:.1f}%)")
            fpy = pass_n / inspected * 100
            self._lbl_fpy.setText(f"{fpy:.1f}%")
        else:
            self._lbl_pass.setText("0 (0.0%)")
            self._lbl_fail.setText("0 (0.0%)")
            self._lbl_review.setText("0 (0.0%)")
            self._lbl_fpy.setText("0.0%")

        # Throughput
        if self._inspection_start_time > 0:
            elapsed = time.time() - self._inspection_start_time
            if elapsed > 0:
                tiles_per_min = inspected / elapsed * 60
                self._lbl_throughput.setText(f"{tiles_per_min:.1f} tiles/min")

        # 결함 분포
        dist_parts = [
            f"{cls}: {count}"
            for cls, count in self._defect_distribution.items()
            if count > 0
        ]
        self._lbl_defect_dist.setText(
            " | ".join(dist_parts) if dist_parts else "No defects detected"
        )

    def _update_elapsed(self):
        """1초마다 경과 시간 갱신."""
        if self._inspection_start_time > 0:
            elapsed = int(time.time() - self._inspection_start_time)
            hrs = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            self._lbl_elapsed.setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")

    # ── 경고음 ─────────────────────────────────────────────────

    @staticmethod
    def _play_alert_sound():
        """FAIL 검출 시 경고음 재생 (UI 블로킹 방지를 위해 별도 스레드)."""
        import threading

        def _beep():
            try:
                import winsound
                winsound.Beep(2500, 150)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    # ── 컬러 이미지 경로 탐색 ───────────────────────────────────

    def _find_colored_image(self, binary_filename):
        """이진화 이미지 파일명으로부터 대응하는 _colored.png 경로 반환."""
        if not self._current_folder:
            return None
        folder = Path(self._current_folder)
        stem = Path(binary_filename).stem
        colored_name = f"{stem}_colored.png"
        colored_path = folder / colored_name
        if colored_path.exists():
            return colored_path
        return None

    # ── 이벤트 필터 ────────────────────────────────────────────

    def eventFilter(self, watched, event):
        """필름스트립 뷰포트의 마우스 휠 스크롤을 차단 (스크롤바 드래그는 유지)."""
        if watched is self._filmstrip.viewport() and event.type() == QEvent.Wheel:
            return True
        return super().eventFilter(watched, event)

    # ── 종료 처리 ──────────────────────────────────────────────

    def closeEvent(self, event):
        """앱 종료 시 Worker 스레드 안전 정리."""
        self._elapsed_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
