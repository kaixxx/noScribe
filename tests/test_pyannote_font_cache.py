"""Guard: the diarization worker keeps matplotlib away from the system fonts
and out of the user-wide font cache.

On macOS the import of pyannote died with ``KeyError: '_items'`` before the
pipeline was even loaded, and because the failed cache build is never written
it died again on every run -- speaker detection was simply dead for that user.
The comment above the ``setdefault`` calls in ``pyannote_mp_worker.py`` has
the mechanism and the measurement.

The variables only count if they are in the environment *before* matplotlib
is imported, so the check has to observe the environment at the moment
pyannote is imported, not afterwards.  A stand-in ``pyannote.audio`` package
does that: it raises with the values it saw, and the worker's own error path
delivers the observation through the result queue.  ``torch`` is stubbed too,
so the test stays free of the heavy stack and runs anywhere the suite does.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import appdirs

REPO = str(Path(__file__).resolve().parent.parent)

STANDIN = textwrap.dedent("""
    import os
    raise RuntimeError("fonts=%r dir=%r" % (
        os.environ.get("MPL_IGNORE_SYSTEM_FONTS"), os.environ.get("MPLCONFIGDIR")))
""")

PROBE = textwrap.dedent("""
    import queue
    from noScribe.pyannote_mp_worker import pyannote_proc_entrypoint
    q = queue.Queue()
    pyannote_proc_entrypoint({"audio_path": "unused.wav"}, q)
    print(q.get_nowait()["error"])
""")


def _run(tmp_path, **env_overrides):
    """Run the worker entrypoint in a clean interpreter against a
    ``pyannote.audio`` that reports the environment it was imported under,
    and an empty ``torch`` so the entrypoint reaches that import."""
    (tmp_path / "torch").mkdir()
    # The one torch call before the pyannote import (Intel macOS only).
    (tmp_path / "torch" / "__init__.py").write_text("def set_num_threads(n): pass\n")
    (tmp_path / "pyannote" / "audio").mkdir(parents=True)
    (tmp_path / "pyannote" / "__init__.py").write_text("")
    (tmp_path / "pyannote" / "audio" / "__init__.py").write_text(STANDIN)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), REPO])}
    env.pop("MPL_IGNORE_SYSTEM_FONTS", None)
    env.pop("MPLCONFIGDIR", None)
    env.update(env_overrides)
    proc = subprocess.run([sys.executable, "-c", PROBE], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_system_font_scan_is_requested_off_when_pyannote_is_imported(tmp_path):
    error = _run(tmp_path)
    assert "fonts='1'" in error, error


def test_font_cache_is_private_to_noscribe(tmp_path):
    """The fonts-less cache must not be written to ~/.matplotlib, where every
    other matplotlib of the same version would load it as the full list --
    nor to a directory the user's shell points matplotlib at, which is the
    same problem one step removed."""
    error = _run(tmp_path, MPLCONFIGDIR=str(tmp_path / "users-own-mplconfig"))
    private = os.path.join(appdirs.user_cache_dir("noScribe"), "matplotlib")
    assert f"dir={private!r}" in error, error


def test_a_deliberate_setting_is_left_alone(tmp_path):
    """setdefault, not assignment: someone who wants matplotlib's full font
    list in the worker can ask for it with an empty value."""
    error = _run(tmp_path, MPL_IGNORE_SYSTEM_FONTS="")
    assert "fonts=''" in error, error
