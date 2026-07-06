# PCB 전체 이미지 그리드 맵 뷰어 (검사 현황 오버레이)
import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor,
    QImage, QPixmap,
)

# Color scheme (industry standard traffic light + low alpha to not obscure background)
CELL_COLORS = {
    "PASS":    QColor(0, 200, 0, 50),      # Light green, very transparent
    "FAIL":    QColor(255, 50, 50, 70),     # Light red
    "REVIEW":  QColor(255, 200, 0, 50),     # Light yellow
    "UNINSPECTED": QColor(0, 0, 0, 60),     # Dark overlay for uninspected areas
}

CELL_BORDER_COLORS = {
    "PASS":    QColor(0, 200, 0, 120),
    "FAIL":    QColor(255, 50, 50, 150),
    "REVIEW":  QColor(255, 200, 0, 120),
    "UNINSPECTED": QColor(80, 80, 80, 100),
}

CAMERA_COLOR = QColor(0, 255, 255, 200)  # Cyan for current scan position


class GlobalView(QGraphicsView):
    """PCB 전체 컬러 이미지 위에 10×10 그리드 오버레이를 표시.
    각 셀은 PASS/FAIL/REVIEW/미검사 상태를 색상으로 표시.
    현재 카메라(검사) 위치를 강조 표시.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setInteractive(False)
        self.setFocusPolicy(Qt.NoFocus)

        self._pixmap_item = None
        self._grid_rows = 0
        self._grid_cols = 0
        self._cell_w = 640
        self._cell_h = 640
        self._cell_items = {}        # (row, col) -> QGraphicsRectItem (fill)
        self._cell_border_items = {} # (row, col) -> QGraphicsRectItem (border)
        self._camera_item = None     # QGraphicsRectItem for camera position
        self._grid_line_items = []   # Grid line items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, bgr_array: np.ndarray):
        """컬러 PCB 배경 이미지 설정.
        BGR numpy array → RGB QPixmap 변환.
        bytesPerLine 명시하여 기울어짐 방지.
        """
        self._scene.clear()
        self._pixmap_item = None
        self._cell_items.clear()
        self._cell_border_items.clear()
        self._camera_item = None
        self._grid_line_items.clear()

        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_grid(self, rows: int, cols: int, cell_w: int = 640, cell_h: int = 640):
        """그리드 오버레이 초기화.
        All cells start as UNINSPECTED (dark semi-transparent overlay).
        Draw thin grid lines.
        """
        self._grid_rows = rows
        self._grid_cols = cols
        self._cell_w = cell_w
        self._cell_h = cell_h

        # Remove previous grid / cell items (keep pixmap)
        for item in self._cell_items.values():
            self._scene.removeItem(item)
        for item in self._cell_border_items.values():
            self._scene.removeItem(item)
        for item in self._grid_line_items:
            self._scene.removeItem(item)
        if self._camera_item is not None:
            self._scene.removeItem(self._camera_item)
            self._camera_item = None
        self._cell_items.clear()
        self._cell_border_items.clear()
        self._grid_line_items.clear()

        # --- Grid lines (Z = 1, between pixmap and cell fills) ---
        grid_pen = QPen(QColor(255, 255, 255, 40))
        grid_pen.setWidthF(0.5)
        grid_pen.setCosmetic(True)

        total_w = cols * cell_w
        total_h = rows * cell_h

        # Vertical lines
        for c in range(cols + 1):
            x = c * cell_w
            line = self._scene.addLine(x, 0, x, total_h, grid_pen)
            line.setZValue(1)
            self._grid_line_items.append(line)

        # Horizontal lines
        for r in range(rows + 1):
            y = r * cell_h
            line = self._scene.addLine(0, y, total_w, y, grid_pen)
            line.setZValue(1)
            self._grid_line_items.append(line)

        # --- Cell overlays (Z = 2) ---
        fill_color = CELL_COLORS["UNINSPECTED"]
        border_color = CELL_BORDER_COLORS["UNINSPECTED"]

        border_pen = QPen(border_color)
        border_pen.setWidthF(1)
        border_pen.setCosmetic(True)

        no_pen = QPen(Qt.NoPen)

        for r in range(rows):
            for c in range(cols):
                rect = QRectF(c * cell_w, r * cell_h, cell_w, cell_h)

                # Fill rect
                fill_item = self._scene.addRect(rect, no_pen, QBrush(fill_color))
                fill_item.setZValue(2)
                self._cell_items[(r, c)] = fill_item

                # Border rect
                border_item = self._scene.addRect(rect, border_pen, QBrush(Qt.NoBrush))
                border_item.setZValue(2)
                self._cell_border_items[(r, c)] = border_item

    def update_cell(self, row: int, col: int, verdict: str):
        """셀 상태 업데이트 (PASS/FAIL/REVIEW).
        verdict must be one of 'PASS', 'FAIL', 'REVIEW'.
        Updates both fill color and border color.
        """
        key = (row, col)
        if key not in self._cell_items:
            return

        verdict_upper = verdict.upper()
        fill_color = CELL_COLORS.get(verdict_upper, CELL_COLORS["UNINSPECTED"])
        border_color = CELL_BORDER_COLORS.get(verdict_upper, CELL_BORDER_COLORS["UNINSPECTED"])

        # Update fill
        self._cell_items[key].setBrush(QBrush(fill_color))

        # Update border
        pen = QPen(border_color)
        pen.setWidthF(1)
        pen.setCosmetic(True)
        self._cell_border_items[key].setPen(pen)

    def set_camera_position(self, row: int, col: int):
        """현재 카메라(검사) 위치 강조 표시.
        Cyan dashed border, thicker (3px), no fill.
        Previous camera highlight is removed.
        """
        self.clear_camera()

        if row < 0 or row >= self._grid_rows or col < 0 or col >= self._grid_cols:
            return

        rect = QRectF(col * self._cell_w, row * self._cell_h,
                       self._cell_w, self._cell_h)

        cam_pen = QPen(CAMERA_COLOR)
        cam_pen.setWidthF(3)
        cam_pen.setCosmetic(True)
        cam_pen.setStyle(Qt.DashLine)

        self._camera_item = self._scene.addRect(rect, cam_pen, QBrush(Qt.NoBrush))
        self._camera_item.setZValue(10)

    def clear_camera(self):
        """카메라 위치 표시 제거."""
        if self._camera_item is not None:
            self._scene.removeItem(self._camera_item)
            self._camera_item = None

    def clear_all(self):
        """모든 아이템 초기화."""
        self._scene.clear()
        self._pixmap_item = None
        self._cell_items.clear()
        self._cell_border_items.clear()
        self._camera_item = None
        self._grid_line_items.clear()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        """리사이즈 시 자동 fitInView."""
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
