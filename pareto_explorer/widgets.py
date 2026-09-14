from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel
class ClickableImageLabel(QLabel):
    def __init__(self, on_click):
        super().__init__()
        self._boxes = {}
        self._on_click = on_click
        self._hovered_id = None
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def set_image(self, pixmap, boxes=None):
        self._boxes = boxes or {}
        self._hovered_id = None
        self.setPixmap(pixmap)
        self.adjustSize()

    def clear_image(self, text=""):
        self._boxes = {}
        self._hovered_id = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _pixmap_offset(self):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None, None
        return max(0.0, (self.width() - pm.width()) / 2.0), max(0.0, (self.height() - pm.height()) / 2.0)

    def _hit(self, local_pos):
        pm = self.pixmap()
        x_offset, y_offset = self._pixmap_offset()
        if pm is None or pm.isNull():
            return None
        ix, iy = local_pos.x() - x_offset, local_pos.y() - y_offset
        if not (0 <= ix <= pm.width() and 0 <= iy <= pm.height()):
            return None
        for element_id, (x0, y0, x1, y1) in self._boxes.items():
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                return element_id
        return None

    @staticmethod
    def _local_pos(event):
        return event.position() if hasattr(event, "position") else QPointF(event.pos())

    def mousePressEvent(self, event):
        element_id = self._hit(self._local_pos(event))
        if element_id is not None:
            self._on_click(element_id)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        element_id = self._hit(self._local_pos(event))
        self.setCursor(Qt.PointingHandCursor if element_id is not None else Qt.ArrowCursor)
        if element_id != self._hovered_id:
            self._hovered_id = element_id
            self.update()  # repaint to draw/clear the highlight overlay
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hovered_id is not None:
            self._hovered_id = None
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._hovered_id is None:
            return
        box = self._boxes.get(self._hovered_id)
        if box is None:
            return
        x_offset, y_offset = self._pixmap_offset()
        if x_offset is None:
            return
        x0, y0, x1, y1 = box
        rect = QRectF(x0 + x_offset - 3, y0 + y_offset - 3, (x1 - x0) + 6, (y1 - y0) + 6)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(66, 133, 244, 230), 3))
        painter.setBrush(QColor(66, 133, 244, 60))
        painter.drawRoundedRect(rect, 8, 8)
        painter.end()

class FitImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._full_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)

    def set_full_pixmap(self, pixmap):
        self.setText("")
        self._full_pixmap = pixmap
        self._refit()

    def clear_pixmap(self, text="--"):
        self._full_pixmap = None
        super().setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit()

    def _refit(self):
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        target = self.size()
        if target.width() < 10 or target.height() < 10:
            return
        scaled = self._full_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)
