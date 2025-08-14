"""PyQt6 entry point loading noScribe UI."""
from pathlib import Path
import sys
from PyQt6 import QtWidgets, uic

from controller import (
    handle_audio_file,
    handle_transcript_file,
    start_transcription,
    stop_transcription,
)
from translator import set_language, t


class NoScribeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        base_dir = Path(__file__).resolve().parent.parent
        uic.loadUi(base_dir / "ui" / "noScribe.ui", self)

        # default to English; could be extended to load from settings
        set_language("en")
        self._retranslate_ui()

        style_path = base_dir / "resources" / "style.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as fh:
                self.setStyleSheet(fh.read())

        self.btnAudioFile.clicked.connect(self.select_audio_file)
        self.btnTranscriptFile.clicked.connect(self.select_transcript_file)
        self.btnStart.clicked.connect(self.start_transcription)
        self.btnStop.clicked.connect(self.stop_transcription)

    def append_log(self, message: str) -> None:
        self.txtLog.append(message)

    def _retranslate_ui(self) -> None:
        """Apply translations to UI elements."""
        self.btnAudioFile.setText(t("Audio File"))
        self.btnTranscriptFile.setText(t("Transcript File"))
        self.btnStart.setText(t("Start"))
        self.btnStop.setText(t("Stop"))

    def select_audio_file(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, t("Select Audio File")
        )
        self.append_log(handle_audio_file(file_name))

    def select_transcript_file(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, t("Select Transcript File")
        )
        self.append_log(handle_transcript_file(file_name))

    def start_transcription(self) -> None:
        self.append_log(start_transcription())

    def stop_transcription(self) -> None:
        self.append_log(stop_transcription())


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = NoScribeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
