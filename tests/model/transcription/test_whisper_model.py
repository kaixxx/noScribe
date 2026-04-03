from noScribe import model


def test_whisper_model_collector(tmp_path):
    # Basic test without any additional path. There is only test whether the
    # functions do not throw any exception as it is unclear which models are
    # present in the package directory.
    tmp = model.transcription.WhisperModelCollector()
    tmp.get_names()
    tmp.get_paths()
    assert tmp.get_path_for("doesnotexist") is None

    # Create an own directory and test the implementation. Adding two
    # directories one without `model.bin` and one with this file. This way,
    # only the model/directory with `model.bin` present should appear.
    user_model_dir = tmp_path / "mymodels"
    user_model_dir.mkdir(parents=True)

    model1 = user_model_dir / "model1"
    model1.mkdir()

    model2 = user_model_dir / "model2"
    model2.mkdir()
    (model2 / "model.bin").touch()

    tmp = model.transcription.WhisperModelCollector([user_model_dir])

    result = tmp.get_names()
    assert "model1" not in result
    assert "model2" in result

    result = tmp.get_paths()
    assert model1.absolute() not in result
    assert model2.absolute() in result

    assert tmp.get_path_for("model2") == model2.absolute()
