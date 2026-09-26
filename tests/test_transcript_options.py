from noScribe.jobs import TranscriptionJob
from noScribe.models import ModelRef
import noScribe.main as main


def test_model_name_matches_the_model_selector_label():
    model = ModelRef("ifs-server", "precise")

    assert main.transcription_model_name(
        {"ifs-server:precise": "precise (IfS-Server)"}, model
    ) == "precise (IfS-Server)"


def test_model_name_falls_back_to_backend_model_id():
    model = ModelRef("local-whisper", "custom-model")

    assert main.transcription_model_name({}, model) == "custom-model"


def test_transcript_options_include_the_selected_model(monkeypatch):
    monkeypatch.setattr(
        main,
        "t",
        lambda key, **_values: key,
    )
    job = TranscriptionJob()

    options = main.format_transcript_options(job, "precise (IfS-Server)")

    assert "label_whisper_model precise (IfS-Server)" in options.split(" | ")
