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
        user_whisper_dir: Path,
    ):
        # Check if desired model is in available models.
        if model_name not in whisper_downloader.get_avail_models():
            raise ValueError(f"model '{model}' not available")

        # Print some overview.
        self.view.download_model(model_name)

        # Download desired model.
        whisper_downloader.download(model_name, user_whisper_dir)