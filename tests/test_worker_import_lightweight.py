"""Guard: importing a worker entrypoint module must not drag in the GUI.

The diarization/whisper subprocesses (multiprocessing "spawn") re-import the
noScribe package to reach their entrypoint. If the package __init__ eagerly
imported noScribe.main, every worker child would load tkinter/customtkinter
and PyAV -- and PyAV's bundled FFmpeg next to torchcodec's system FFmpeg
triggers objc duplicate-class warnings on macOS. These tests run in a clean
interpreter so the parent process's own imports can't mask a regression.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)


def _run(code):
    env = {**os.environ, "PYTHONPATH": REPO}
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)


def test_worker_import_pulls_no_gui_modules():
    res = _run(
        "import sys; import noScribe.pyannote_mp_worker; "
        "heavy = [m for m in ('av', 'tkinter', 'customtkinter', 'noScribe.main') "
        "if m in sys.modules]; "
        "sys.exit('GUI modules leaked into worker import: %s' % heavy if heavy else 0)"
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_lazy_main_attribute_still_works():
    # __main__.py does noScribe.main.noScribeMain() -- the lazy __getattr__
    # must keep that path alive.
    res = _run(
        "import noScribe; "
        "assert callable(noScribe.main.noScribeMain), 'noScribeMain missing'"
    )
    assert res.returncode == 0, res.stdout + res.stderr
