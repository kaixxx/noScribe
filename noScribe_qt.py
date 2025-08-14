import sys
from PyQt6 import QtWidgets
from ui.noScribe_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.startButton.clicked.connect(self.handle_start)
        self.stopButton.clicked.connect(self.handle_stop)

    def handle_start(self):
        self.logText.append("Start clicked")

    def handle_stop(self):
        self.logText.append("Stop clicked")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
