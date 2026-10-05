from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

from noScribe import main


def test_frozen_macos_uses_separately_installed_editor(monkeypatch):
    """The frozen app must not import the editor as a bundled resource."""
    calls = []
    app = SimpleNamespace(_headless=True)

    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        main.impres,
        "files",
        lambda *_: pytest.fail("frozen macOS must not look for a bundled editor"),
    )
    monkeypatch.setattr(main.os.path, "exists", lambda path: True)
    monkeypatch.setattr(main, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))

    main.App.launch_editor(app, "/tmp/transcript.html")

    editor = Path("/Applications/noScribeEdit.app/Contents/MacOS/noScribeEdit")
    assert calls == [([editor, "/tmp/transcript.html"], {"start_new_session": True})]
