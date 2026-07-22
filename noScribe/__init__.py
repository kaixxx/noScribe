"""noScribe package.

Importing this package must stay lightweight: the diarization/whisper worker
subprocesses (multiprocessing "spawn") re-import it just to reach their
entrypoint module. An eager ``from noScribe import main`` would drag the whole
GUI stack (tkinter, customtkinter, PyAV with its bundled FFmpeg) into every
worker child -- and on macOS, PyAV's FFmpeg loaded next to torchcodec's system
FFmpeg triggers objc duplicate-class warnings. ``noScribe.main`` is therefore
loaded lazily (PEP 562); ``__main__.py``'s ``noScribe.main.noScribeMain()``
keeps working unchanged.
"""
import importlib


def __getattr__(name):
    if name == "main":
        return importlib.import_module("noScribe.main")
    raise AttributeError(f"module 'noScribe' has no attribute {name!r}")
