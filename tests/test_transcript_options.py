from noScribe.jobs import TranscriptionJob
from noScribe.models import ModelDescriptor, ModelRef
import noScribe.main as main


def test_model_name_matches_the_model_selector_label():
    model = ModelRef("ifs-server", "precise")

    assert main.transcription_model_name(
        {"ifs-server:precise": "precise (IfS-Server)"}, model
    ) == "precise (IfS-Server)"


def test_model_name_falls_back_to_backend_model_id():
    model = ModelRef("local-whisper", "custom-model")

    assert main.transcription_model_name({}, model) == "custom-model"


def test_transcription_model_catalog_initializes_all_ui_mappings():
    model = ModelDescriptor(
        ref=ModelRef("local-whisper", "precise"),
        display_name="precise",
        engine="test",
        capabilities=frozenset({"transcription"}),
    )

    class Registry:
        def list_models(self, capability):
            assert capability == "transcription"
            return [model]

        def model_options(self, capability):
            assert capability == "transcription"
            return {"precise": model}

    class App:
        inference_backend = Registry()

    app = App()
    main._update_transcription_model_catalog(app)

    assert app.transcription_models == {"local-whisper:precise": model}
    assert app.transcription_model_options == {"precise": model}
    assert app.transcription_model_labels == {"local-whisper:precise": "precise"}


def test_transcript_options_include_the_selected_model(monkeypatch):
    monkeypatch.setattr(
        main,
        "t",
        lambda key, **_values: key,
    )
    job = TranscriptionJob()

    options = main.format_transcript_options(job, "precise (IfS-Server)")

    assert "label_whisper_model precise (IfS-Server)" in options.split(" | ")
