# QGraphicsView 기반 고해상도 결함 뷰어 (줌/패닝/코너 브라켓/브러쉬 드로잉)

import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QImage, QPixmap, QFont, QFontMetricsF,
)


def confidence_color(
    conf: float,
    cls_id: int = -1,
    per_class_bands: dict | None = None,
) -> QColor:
    """클래스별 REVIEW 밴드 기반 색상 반환.

    per_class_bands가 주어지면 해당 클래스의 (review_min, review_max)를 기준으로:
      conf > review_max  → Red (FAIL 수준)
      conf >= review_min → Yellow (REVIEW 수준)
      conf < review_min  → Gray (PASS 수준)

    per_class_bands가 없으면 레거시 고정 임계값(0.70/0.90) 사용.
    """
    if per_class_bands and cls_id in per_class_bands:
        r_min, r_max = per_class_bands[cls_id]
        if conf > r_max:
            return QColor(255, 68, 68)
        elif conf >= r_min:
            return QColor(255, 215, 0)
        else:
            return QColor(136, 136, 136)
    # 레거시 폴백
    if conf >= 0.90:
        return QColor(255, 68, 68)
    elif conf >= 0.70:
        return QColor(255, 215, 0)
    return QColor(136, 136, 136)


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
    """고해상도 결함 확대 뷰어 (Local View).

    기능:
    - 마우스 휠: 포인터 중심 확대/축소 (scale factor 1.15)
    - 우클릭 드래그: 패닝
    - 좌클릭 드래그: 브러쉬 드로잉 (흰색으로 이미지 마스킹)
    - 코너 브라켓 렌더링 (Cosmetic Pen → 줌 불변 두께 2~3px)
    - 신호등 배색 (confidence 기반)
    - Shift 홀드: 오버레이 숨김 토글
    - 브러쉬 커서 프리뷰 (반투명 시안 원)
    """

    image_edited = pyqtSignal(object)
    brush_size_changed = pyqtSignal(int)
    # 마우스 씬 좌표를 MainWindow에 전달 (GlobalView 커서 동기화용)
    cursor_moved = pyqtSignal(float, float)   # scene_x, scene_y
    cursor_left = pyqtSignal()                # 마우스가 뷰어를 떠남

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setMouseTracking(True)  # 클릭 없이도 마우스 이동 추적

        self._pixmap_item = None
        self._overlay_items = []
        self._highlighted_overlay_items = []
        self._other_overlay_items = []
        self._overlay_visible = True
        self._zoom_factor = 1.15
        self._min_zoom = 0.1      # 최소 10% 축소 허용
        self._max_zoom = 50.0     # 최대 50배 확대 허용 (픽셀 단위 검사 가능)
        self._panning = False
        self._pan_start = QPointF()

        # 브러쉬 드로잉 상태
        self._cv_image: np.ndarray | None = None
        self._brush_size: int = 10
        self._drawing: bool = False
        self._last_draw_pt: tuple | None = None  # (x, y) 정수 튜플

        # 브러쉬 커서 프리뷰 (반투명 시안 원)
        self._cursor_item = None

    # ── Public API ─────────────────────────────────────────────

    def set_image(self, bgr_array: np.ndarray):
        """BGR numpy 배열을 scene에 표시. 기존 오버레이는 제거됩니다.

        bytesPerLine을 명시하여 BGR→RGB 변환 시
        이미지 기울어짐/찌그러짐 방지.
        전달받은 배열의 참조를 유지하여 브러쉬 드로잉 시 직접 수정합니다.
        """
        self._clear_overlay()
        self._remove_cursor_item()
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None

        # 원본 ndarray 참조 저장 (브러쉬 드로잉 시 직접 수정 대상)
        self._cv_image = bgr_array

        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # .copy()로 numpy 버퍼에서 분리 — numpy GC 후에도 안전
        pixmap = QPixmap.fromImage(qimg.copy())

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def set_detections(self, detections, highlight_index=-1, per_class_bands=None):
        """결함 리스트로 코너 브라켓 + 라벨 오버레이 렌더링.

        Args:
            detections: bbox_abs, class_name, confidence, class_id 포함 dict 리스트.
            highlight_index: 강조 표시할 결함 인덱스 (더 두꺼운 선).
            per_class_bands: {class_id: (review_min, review_max)} — 색상 결정용.
        """
        self._clear_overlay()
        self._highlighted_overlay_items = []
        self._other_overlay_items = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox_abs"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            cls_id = det.get("class_id", -1)
            color = confidence_color(conf, cls_id=cls_id, per_class_bands=per_class_bands)

            is_highlighted = (i == highlight_index)
            thickness = 3.0 if is_highlighted else 2.0

            bracket_items = self._draw_corner_brackets(
                x1, y1, x2, y2, color, thickness
            )

            label_text = f"{cls_name} {conf:.2f}"
            label_item = DefectLabel(label_text, color)
            label_item.setPos(x1, y1 - 2)
            self._scene.addItem(label_item)
            
            items_for_this_defect = bracket_items + [label_item]
            self._overlay_items.extend(items_for_this_defect)
            
            if is_highlighted:
                self._highlighted_overlay_items.extend(items_for_this_defect)
            else:
                self._other_overlay_items.extend(items_for_this_defect)

        self.set_overlay_visible(self._overlay_visible)

    def set_overlay_visible(self, visible: bool):
        """Shift 홀드 시 오버레이 숨김 토글 (강조된 결함은 항상 표시)."""
        self._overlay_visible = visible
        for item in self._highlighted_overlay_items:
            item.setVisible(True)
        for item in self._other_overlay_items:
            item.setVisible(visible)

    def zoom_to_rect(self, x1, y1, x2, y2, pad_ratio=0.5):
        """특정 영역으로 줌. 패딩을 추가하여 주변 컨텍스트를 노출."""
        w, h = x2 - x1, y2 - y1
        pad_x = w * pad_ratio
        pad_y = h * pad_ratio
        rect = QRectF(
            x1 - pad_x, y1 - pad_y,
            w + 2 * pad_x, h + 2 * pad_y,
        )
        self.fitInView(rect, Qt.KeepAspectRatio)

    def clear_all(self):
        """Scene 전체 초기화."""
        self._scene.clear()
        self._pixmap_item = None
        self._overlay_items = []
        self._highlighted_overlay_items = []
        self._other_overlay_items = []
        self._cv_image = None
        self._cursor_item = None

    def zoom(self, zoom_in: bool):
        """키보드 단축키용 줌인/줌아웃 (중앙 기준)."""
        factor = self._zoom_factor if zoom_in else (1.0 / self._zoom_factor)
        self._apply_zoom(factor)

    def set_brush_size(self, size: int):
        """외부에서 브러쉬 크기를 설정 (MainWindow 단축키용)."""
        self._brush_size = max(1, min(100, size))

    def _apply_zoom(self, factor: float):
        """줌 배율 적용 시 설정된 최소/최대 한계를 벗어나지 않도록 제한합니다."""
        current_zoom = self.transform().m11()
        new_zoom = current_zoom * factor

        if new_zoom < self._min_zoom:
            factor = self._min_zoom / current_zoom
        elif new_zoom > self._max_zoom:
            factor = self._max_zoom / current_zoom

        if factor != 1.0:
            self.scale(factor, factor)

    def pan(self, dx: int, dy: int):
        """키보드 단축키용 패닝."""
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        h_bar.setValue(h_bar.value() + dx)
        v_bar.setValue(v_bar.value() + dy)

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

    # ── Brush Cursor Preview ───────────────────────────────────

    def _update_cursor_preview(self, scene_pt: QPointF):
        """브러쉬 크기를 나타내는 반투명 원을 씬 좌표에 표시.

        산업용 검사 소프트웨어에서 사용하는 시안(Cyan) 계열의
        반투명 원으로, 흰색 배경과 어두운 배경 모두에서 잘 보입니다.
        """
        r = self._brush_size / 2.0
        x, y = scene_pt.x(), scene_pt.y()

        # orphan 아이템 감지: scene.clear() 후 참조가 남아있으면 폐기
        if self._cursor_item is not None and self._cursor_item.scene() is None:
            self._cursor_item = None

        if self._cursor_item is not None:
            # 위치만 갱신 (생성/삭제 비용 절약)
            self._cursor_item.setRect(x - r, y - r, r * 2, r * 2)
        else:
            pen = QPen(QColor(0, 220, 255, 180), 1.5)
            pen.setCosmetic(True)
            brush = QBrush(QColor(0, 220, 255, 40))
            self._cursor_item = self._scene.addEllipse(
                x - r, y - r, r * 2, r * 2, pen, brush
            )
            self._cursor_item.setZValue(1000)  # 최상위 레이어

    def _remove_cursor_item(self):
        """브러쉬 커서 아이템 제거."""
        if self._cursor_item is not None:
            if self._cursor_item.scene() is not None:
                self._scene.removeItem(self._cursor_item)
            self._cursor_item = None

    # ── Brush Drawing ──────────────────────────────────────────

    def _refresh_pixmap(self):
        """cv_image로부터 QPixmap을 재생성하여 배경만 교체.

        오버레이(브라켓, 라벨)는 별도 아이템이므로 영향 없음.
        """
        if self._cv_image is None or self._pixmap_item is None:
            return
        rgb = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _scene_to_pixel(self, viewport_pos) -> tuple | None:
        """뷰포트 좌표 → 이미지 픽셀 좌표(정수) 변환.

        이미지 범위 밖이면 None 반환.
        """
        if self._cv_image is None:
            return None
        scene_pt = self.mapToScene(viewport_pos)
        x, y = int(scene_pt.x()), int(scene_pt.y())
        h, w = self._cv_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    # ── Internal ───────────────────────────────────────────────

    def _clear_overlay(self):
        """기존 오버레이 아이템 일괄 제거."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        self._highlighted_overlay_items = []
        self._other_overlay_items = []

    # ── Event Handlers ─────────────────────────────────────────

    def wheelEvent(self, event):
        """마우스 포인터 중심 확대/축소."""
        angle = event.angleDelta().y()
        if angle > 0:
            factor = self._zoom_factor
        elif angle < 0:
            factor = 1.0 / self._zoom_factor
        else:
            return
        self._apply_zoom(factor)

    def mousePressEvent(self, event):
        """좌클릭: 브러쉬 드로잉 시작 / 우클릭: 패닝 시작."""
        if event.button() == Qt.LeftButton:
            pt = self._scene_to_pixel(event.pos())
            if pt is not None:
                self._drawing = True
                self._last_draw_pt = pt
                # 클릭 지점에 점 하나 찍기 (드래그 없이 클릭만 해도 표시)
                cv2.circle(
                    self._cv_image, pt,
                    self._brush_size // 2, (255, 255, 255), -1
                )
                self._refresh_pixmap()
            event.accept()
        elif event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self._remove_cursor_item()  # 우클릭 패닝 시작 시 커서 프리뷰 숨김(잔상 방지)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """드로잉 중: cv2.line으로 실시간 그리기 / 패닝 중: 뷰 이동 / 기본: 커서 프리뷰."""
        scene_pt = self.mapToScene(event.pos())

        if self._drawing:
            pt = self._scene_to_pixel(event.pos())
            if pt is not None and self._last_draw_pt is not None:
                cv2.line(
                    self._cv_image,
                    self._last_draw_pt, pt,
                    (255, 255, 255),
                    self._brush_size,
                )
                self._last_draw_pt = pt
                self._refresh_pixmap()
            # 드로잉 중에도 커서 프리뷰 + 위치 시그널 발생
            self._update_cursor_preview(scene_pt)
            self.cursor_moved.emit(scene_pt.x(), scene_pt.y())
            event.accept()
        elif self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
        else:
            # 기본 상태: 커서 프리뷰 표시 + 위치 시그널 발생
            self._update_cursor_preview(scene_pt)
            self.cursor_moved.emit(scene_pt.x(), scene_pt.y())
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """드로잉 완료 → image_edited 시그널 발생 / 패닝 종료."""
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._last_draw_pt = None
            if self._cv_image is not None:
                self.image_edited.emit(self._cv_image)
            event.accept()
        elif event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """마우스가 뷰어 영역을 벗어나면 커서 프리뷰 제거."""
        self._remove_cursor_item()
        self.cursor_left.emit()
        super().leaveEvent(event)
