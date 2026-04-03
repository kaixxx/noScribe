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