# PCB 실시간 검사용 읽기 전용 타일 뷰어 (줌/패닝/브러쉬 없음)

import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QImage, QPixmap, QFont, QFontMetricsF,
)


def confidence_color(
    conf: float,
    class_name: str | None = None,
    per_class_bands: dict | None = None,
) -> QColor:
    """클래스별 REVIEW 밴드 기반 색상 반환.

    per_class_bands가 주어지면 해당 클래스의 (review_min, review_max)를 기준으로:
      conf > review_max  → Red (FAIL 수준)
      conf >= review_min → Yellow (REVIEW 수준)
      conf < review_min  → Green (PASS 수준)

    per_class_bands가 없으면 레거시 고정 임계값(0.70/0.90) 사용.
    """
    if per_class_bands and class_name in per_class_bands:
        r_min, r_max = per_class_bands[class_name]
        if conf > r_max:
            return QColor(255, 68, 68)
        elif conf >= r_min:
            return QColor(255, 215, 0)
        else:
            return QColor(0, 200, 0)
    # 레거시 폴백
    if conf >= 0.90:
        return QColor(255, 68, 68)
    elif conf >= 0.70:
        return QColor(255, 215, 0)
    return QColor(0, 200, 0)


class DefectLabel(QGraphicsItem):
    """결함 클래스명 + 신뢰도 라벨 (줌 불변 크기, 70% 불투명도 배경).

    ItemIgnoresTransformations 플래그로 줌 레벨에 관계없이
    화면상 동일 크기를 유지합니다.
    """

    def __init__(self, text, color, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self._text = text
        self._color = color
        self._font = QFont("Consolas", 9)
        self._font.setBold(True)
        fm = QFontMetricsF(self._font)
        self._text_rect = fm.boundingRect(self._text)
        self._padding = 4

    def boundingRect(self):
        r = self._text_rect
        return QRectF(
            r.x() - self._padding,
            r.y() - self._padding,
            r.width() + 2 * self._padding,
            r.height() + 2 * self._padding,
        )

    def paint(self, painter, option, widget):
        rect = self.boundingRect()
        # 70% 불투명도(0.7 × 255 ≈ 178) 어두운 배경
        painter.fillRect(rect, QColor(0, 0, 0, 178))
        painter.setFont(self._font)
        painter.setPen(QPen(self._color))
        painter.drawText(rect, Qt.AlignCenter, self._text)


class VisionViewer(QGraphicsView):
    """읽기 전용 타일 뷰어. 이미지 표시와 결함 오버레이만 수행.
    브러쉬, 줌, 패닝, 마우스 이벤트 일절 없음.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFocusPolicy(Qt.NoFocus)
        self.setInteractive(False)  # 완전 비대화형
        self._pixmap_item = None
        self._overlay_items = []

    # ── Public API ─────────────────────────────────────────────

    def set_image(self, bgr_array: np.ndarray):
        """BGR numpy 배열을 scene에 표시. 기존 오버레이 제거.

        bytesPerLine 명시하여 이미지 기울어짐 방지.
        Auto-fit to view.
        """
        self._clear_overlay()
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None

        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # .copy()로 numpy 버퍼에서 분리 — numpy GC 후에도 안전
        pixmap = QPixmap.fromImage(qimg.copy())

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_detections(self, detections: list, per_class_bands: dict | None = None):
        """결함 리스트로 코너 브라켓 + 라벨 오버레이 렌더링.

        Args:
            detections: list of dicts with bbox_abs, class_name, confidence
            per_class_bands: {class_name: (review_min, review_max)} — 색상 결정용.
        All detections shown with equal thickness (2.5px cosmetic pen).
        """
        self._clear_overlay()

        for det in detections:
            x1, y1, x2, y2 = det["bbox_abs"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            color = confidence_color(conf, class_name=cls_name, per_class_bands=per_class_bands)

            bracket_items = self._draw_corner_brackets(
                x1, y1, x2, y2, color, thickness=2.5
            )

            label_text = f"{cls_name} {conf:.2f}"
            label_item = DefectLabel(label_text, color)
            label_item.setPos(x1, y1 - 2)
            self._scene.addItem(label_item)

            self._overlay_items.extend(bracket_items + [label_item])

    def clear_all(self):
        """Scene 전체 초기화."""
        self._scene.clear()
        self._pixmap_item = None
        self._overlay_items = []

    # ── Event Handlers ─────────────────────────────────────────

    def resizeEvent(self, event):
        """리사이즈 시 자동 fitInView."""
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ── Corner Bracket Rendering ───────────────────────────────

    def _draw_corner_brackets(self, x1, y1, x2, y2, color,
                               thickness=2.5, arm_ratio=0.2):
        """네 모서리의 직각 꺾쇠( ┌ ┐ └ ┘ )만 렌더링.

        Cosmetic Pen(setCosmetic(True))으로 줌 레벨에 관계없이
        화면상 2~3px 일정 두께를 유지합니다.
        닫힌 Bounding Box는 PCB 미세 회로를 가려 판독을 방해하므로 금지.
        """
        pen = QPen(color, thickness)
        pen.setCosmetic(True)

        w, h = x2 - x1, y2 - y1
        arm_x = w * arm_ratio
        arm_y = h * arm_ratio

        corners = [
            # ┌ Top-Left
            [(x1, y1 + arm_y), (x1, y1), (x1 + arm_x, y1)],
            # ┐ Top-Right
            [(x2 - arm_x, y1), (x2, y1), (x2, y1 + arm_y)],
            # └ Bottom-Left
            [(x1, y2 - arm_y), (x1, y2), (x1 + arm_x, y2)],
            # ┘ Bottom-Right
            [(x2 - arm_x, y2), (x2, y2), (x2, y2 - arm_y)],
        ]

        items = []
        for pts in corners:
            path = QPainterPath()
            path.moveTo(pts[0][0], pts[0][1])
            path.lineTo(pts[1][0], pts[1][1])
            path.lineTo(pts[2][0], pts[2][1])
            item = self._scene.addPath(path, pen)
            items.append(item)
        return items

    # ── Internal ───────────────────────────────────────────────

    def _clear_overlay(self):
        """기존 오버레이 아이템 일괄 제거."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
