"""Guard: the diarization worker can hand the bundled pipeline folder to
pyannote on every Python version the project supports.

``pyannote`` is a namespace shared by the repository's ``pyannote/`` folder
(config.yaml and the models) and the installed library, so
``impres.files("pyannote")`` is a view over both directories. The stdlib's
``importlib.resources.as_file`` only accepts a directory from Python 3.12 on;
on 3.10 and 3.11 every diarization failed at once with
``FileNotFoundError: MultiplexedPath(...) is not a file``. dce1150 switched
main.py and the Whisper worker to the ``importlib_resources`` backport there,
but not this worker. CI runs 3.10, so this test fails there if the worker goes
back to the stdlib module.
"""
from noScribe import pyannote_mp_worker


def test_bundled_pipeline_folder_reaches_pyannote():
    impres = pyannote_mp_worker.impres
    with impres.as_file(impres.files("pyannote")) as path:
        assert (path / "config.yaml").is_file()
