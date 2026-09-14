import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from theme import _light_palette


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
