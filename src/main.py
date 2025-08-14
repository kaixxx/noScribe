"""PyQt6 entry point loading noScribe UI."""
from pathlib import Path
import sys
from PyQt6 import QtWidgets, uic

from controller import NoScribeController
from translator import t


class NoScribeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        base_dir = Path(__file__).resolve().parent.parent
        uic.loadUi(base_dir / "ui" / "noScribe.ui", self)

        self.controller = NoScribeController()
        self.controller.log.connect(self.append_log)

        self._retranslate_ui()

        style_path = base_dir / "resources" / "style.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as fh:
                self.setStyleSheet(fh.read())

        self.btnAudioFile.clicked.connect(self.select_audio_file)
        self.btnTranscriptFile.clicked.connect(self.select_transcript_file)
        self.btnStart.clicked.connect(self.controller.start_transcription)
        self.btnStop.clicked.connect(self.controller.stop_transcription)
        self.cmbLanguage.currentIndexChanged.connect(self.change_language)
        self.chkDiarization.toggled.connect(self.controller.set_diarization)

    def append_log(self, message: str) -> None:
        self.txtLog.append(message)

    def _retranslate_ui(self) -> None:
        """Apply translations to UI elements."""
        self.btnAudioFile.setText(t("Audio File"))
        self.btnTranscriptFile.setText(t("Transcript File"))
        self.btnStart.setText(t("Start"))
        self.btnStop.setText(t("Stop"))
        self.lblLanguage.setText(t("Language"))
        self.chkDiarization.setText(t("Diarization"))
        self.cmbLanguage.blockSignals(True)
        current = self.controller.language
        self.cmbLanguage.clear()
        self.cmbLanguage.addItem(t("English"), "en")
        self.cmbLanguage.addItem(t("German"), "de")
        index = self.cmbLanguage.findData(current)
        if index != -1:
            self.cmbLanguage.setCurrentIndex(index)
        self.cmbLanguage.blockSignals(False)

    def select_audio_file(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, t("Select Audio File")
        )
        self.controller.handle_audio_file(file_name)

    def select_transcript_file(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, t("Select Transcript File")
        )
        self.controller.handle_transcript_file(file_name)

    def change_language(self) -> None:
        data = self.cmbLanguage.currentData()
        if data:
            self.controller.set_language(data)
            self._retranslate_ui()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = NoScribeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
