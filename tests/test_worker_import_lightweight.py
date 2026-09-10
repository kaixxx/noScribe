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

import pytest

REPO = str(Path(__file__).resolve().parent.parent)

# Both are ctx.Process targets, so both are re-imported in a spawn child.
WORKER_MODULES = ["noScribe.pyannote_mp_worker", "noScribe.whisper_mp_worker"]


def _without_comments(text):
    """Source with comment lines dropped. Both checks below look for a literal
    that these files also *explain* in a comment; matching the explanation
    would let them pass while the thing itself was gone."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _run(code, home=None):
    env = {**os.environ, "PYTHONPATH": REPO}
    if home is not None:
        # noScribe.main creates its config directory and writes config.yml at
        # import time; point that at the test's own directory so the suite
        # never touches the real one.
        env.update(HOME=str(home), APPDATA=str(home), XDG_CONFIG_HOME=str(home))
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)


@pytest.mark.parametrize("module", WORKER_MODULES)
def test_worker_import_pulls_no_gui_modules(module):
    res = _run(
        f"import sys; import {module}; "
        "heavy = [m for m in ('av', 'tkinter', 'customtkinter', 'noScribe.main') "
        "if m in sys.modules]; "
        "sys.exit('GUI modules leaked into worker import: %s' % heavy if heavy else 0)"
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_every_spec_declares_the_lazy_main_import():
    # PyInstaller's static analysis cannot follow importlib.import_module, so a
    # spec that does not name noScribe.main produces a build that dies at
    # startup with ModuleNotFoundError -- which no other test would catch.
    specs = sorted((Path(REPO) / "pyinstaller").glob("noScribe_*.spec"))
    assert specs, "no noScribe pyinstaller specs found"
    missing = []
    for spec in specs:
        code = _without_comments(spec.read_text())
        if "'noScribe.main'" not in code and '"noScribe.main"' not in code:
            missing.append(spec.name)
    assert not missing, f"specs missing the noScribe.main hidden import: {missing}"


def test_lazy_main_attribute_still_works(tmp_path):
    # __main__.py does noScribe.main.noScribeMain() -- the lazy __getattr__
    # must keep that path alive.
    res = _run(
        "import noScribe; "
        "assert callable(noScribe.main.noScribeMain), 'noScribeMain missing'",
        home=tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_main_stays_discoverable_without_importing_it():
    # A lazy attribute is invisible to dir() and to star-import unless the
    # module says otherwise -- and looking it up must not trigger the import it
    # avoids, nor list the name twice once it has been bound.
    res = _run(
        "import sys, noScribe; "
        "assert dir(noScribe).count('main') == 1, dir(noScribe); "
        "assert 'noScribe.main' not in sys.modules, 'dir() triggered the import'; "
        "ns = {}; exec('from noScribe import *', ns); "
        "assert 'main' in ns, 'star-import no longer binds main'"
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_submodules_stay_reachable_from_the_package():
    """main.py used to bind these on the package as a side effect of being
    imported eagerly; dropping that would turn `noScribe.utils` into an
    AttributeError for a module that exists."""
    res = _run(
        "import noScribe; "
        "assert noScribe.utils.ms_to_str(1000) == '00:00:01', 'utils unreachable'"
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_freeze_support_runs_before_main_is_imported():
    """Under PyInstaller a spawn child is the frozen binary started again, so
    it re-runs __main__.py and freeze_support() has to intercept it there.
    main.py calls it too, but only after importing tkinter, customtkinter and
    PyAV -- the very stack a worker child must not load."""
    source = _without_comments((Path(REPO) / "noScribe" / "__main__.py").read_text())
    assert "mp.freeze_support()" in source
    assert source.index("mp.freeze_support()") < source.index("noScribe.main"), \
        "freeze_support() must come before anything touches noScribe.main"
