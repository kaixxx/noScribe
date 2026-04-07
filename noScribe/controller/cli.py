from pathlib import Path

from noScribe import model, view


class Controller:
    def __init__(
        self,
        cli_view: view.CommandLine,
    ):
        self.view = cli_view

    def print_available_whisper_models(
        self, whisper_collector: model.transcription.WhisperModelCollector
    ):
        # Get all whisper models.
        model_names = whisper_collector.get_names()

        # Show in view.
        self.view.show_available_models(model_names)

    def download_model_files(
        self,
        whisper_downloader: model.transcription.WhisperModelDownloader,
        model_name: str,
    ):
        # Print some overview.
        self.view.download_model(model_name)

        # Download desired model.
        try:
            whisper_downloader.download(model=model_name, force=False)
        except model.transcription.exception.ModelAlreadyExists:
            self.view.download_model_failed_already_exists(model_name)
