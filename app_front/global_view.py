# PCB 전체 이미지 뷰어 (현재 검사 위치를 실제 픽셀 좌표 기반 박스 1개로 표시)
import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor,
    QImage, QPixmap,
)

SCAN_BOX_COLOR = QColor(255, 40, 40, 230)  # 빨간색 — 현재 검사 위치


class GlobalView(QGraphicsView):
    """PCB 전체 컬러 이미지 위에 현재 검사 위치를 빨간 박스 1개로 표시.

    오버랩 설정에 따라 타일이 서로 겹칠 수 있어 그리드 셀 단위의 PASS/FAIL/REVIEW
    색상 표시는 더 이상 의미가 없다 — 실제 타일 위치(픽셀 좌표)만 정확히 보여준다.
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
        self._scan_box_item = None  # QGraphicsRectItem, 현재 검사 위치

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
        self._scan_box_item = None

        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_scan_box(self, x: float, y: float, size: float):
        """현재 검사 중인 타일 위치를 실제 픽셀 좌표 기준으로 표시.

        x, y: 타일 좌상단의 이미지 픽셀 좌표 (오버랩 스트라이드 반영된 실제 위치)
        size: 타일 한 변의 길이(px, 정사각형)
        """
        if x < 0 or y < 0:
            self.clear_scan_box()
            return

        rect = QRectF(x, y, size, size)

        pen = QPen(SCAN_BOX_COLOR)
        pen.setWidthF(3)
        pen.setCosmetic(True)

        if self._scan_box_item is not None:
            self._scan_box_item.setRect(rect)
        else:
            self._scan_box_item = self._scene.addRect(rect, pen, QBrush(Qt.NoBrush))
            self._scan_box_item.setZValue(10)

    def clear_scan_box(self):
        """검사 위치 박스 제거."""
        if self._scan_box_item is not None:
            self._scene.removeItem(self._scan_box_item)
            self._scan_box_item = None

    def clear_all(self):
        """모든 아이템 초기화."""
        self._scene.clear()
        self._pixmap_item = None
        self._scan_box_item = None

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        """리사이즈 시 자동 fitInView."""
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
