"""noScribe package.

Importing this package must stay lightweight: the diarization/whisper worker
subprocesses (multiprocessing "spawn") re-import it just to reach their
entrypoint module. An eager ``from noScribe import main`` would drag the whole
GUI stack (tkinter, customtkinter, PyAV with its bundled FFmpeg) into every
worker child -- and on macOS, PyAV's FFmpeg loaded next to torchcodec's system
FFmpeg triggers objc duplicate-class warnings. The submodules are therefore
loaded lazily (PEP 562); ``__main__.py``'s ``noScribe.main.noScribeMain()``
keeps working unchanged.

PyInstaller cannot follow these imports, so every spec in pyinstaller/ lists
``noScribe.main`` in its hiddenimports -- without that the frozen app dies at
startup with ModuleNotFoundError.
"""
import importlib as _importlib

# Everything main.py used to bind on the package as a side effect of being
# imported eagerly. Kept so `import noScribe; noScribe.utils...` still works
# and so a typo gets an honest "no attribute" rather than a stale one.
_SUBMODULES = ("main", "audio", "exception", "transcription", "utils")

# Star-import reads __all__, never __dir__; without this `from noScribe import *`
# would bind the helper names above instead of the package's own module.
__all__ = ["main"]


def __getattr__(name):
    if name in _SUBMODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # A set: the import system binds each submodule into globals() on first
    # access, so a list would show it twice from then on.
    return sorted({*globals(), *_SUBMODULES})
