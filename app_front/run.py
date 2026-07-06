# PCB 실시간 검사 모니터링 프로그램 진입점

import sys
from pathlib import Path

# PyTorch를 PyQt5보다 먼저 import (Windows DLL 충돌 방지)
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

from main_ui import MainWindow


def apply_dark_theme(app: QApplication):
    """Fusion 스타일 + 산업용 다크 팔레트 적용."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, QColor(200, 200, 200))
    palette.setColor(QPalette.Text, QColor(200, 200, 200))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(240, 240, 240))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(100, 100, 100))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 100))
    app.setPalette(palette)

    app.setFont(QFont("Segoe UI", 9))


def main():
    # High DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
