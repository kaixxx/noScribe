"""Pins the initialfile each file dialog gets; the reasons are in the comments in
`button_audio_file_event`. The one that matters most: Tk's own dialog under X11
appends each confirmed file to -initialfile instead of replacing it (tkfbox.tcl,
ActivateEnt -> VerifyFileName: `lappend data(selectFile) $file`), so pre-selecting
the current files there made a file once picked impossible to deselect (#340).
"""
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

from noScribe import main


def _dialog_options(monkeypatch, windowing_system, audio_files):
    seen = {}

    def askopenfilename(**kwargs):
        seen.update(kwargs)
        return ''  # cancelled: the handler returns before touching the GUI

    def tk_call(*args):
        assert args == ('tk', 'windowingsystem')
        return windowing_system

    monkeypatch.setattr(main.tk.filedialog, 'askopenfilename', askopenfilename)
    app = SimpleNamespace(audio_files_list=audio_files, tk=SimpleNamespace(call=tk_call))
    main.App.button_audio_file_event(app)
    return seen['initialfile'], seen['initialdir']


@pytest.fixture
def audio(tmp_path):
    """Real files, because the macOS branch only names a file that exists."""
    names = ['Interview #1.m4a', 'interview_02.m4a']
    for name in names:
        (tmp_path / name).touch()
    return [str(tmp_path / name) for name in names]


@pytest.mark.parametrize('windowing_system, pick, initialfile', [
    ('x11', 1, ''),
    ('x11', 2, ''),
    ('aqua', 1, 'Interview%20%231.m4a'),
    ('aqua', 2, ''),
    ('win32', 1, '"Interview #1.m4a"'),
    ('win32', 2, '"Interview #1.m4a" "interview_02.m4a"'),
], ids=['x11-one', 'x11-batch', 'aqua-one', 'aqua-batch', 'win32-one', 'win32-batch'])
def test_preselection_per_dialog(monkeypatch, audio, windowing_system, pick, initialfile):
    assert _dialog_options(monkeypatch, windowing_system, audio[:pick]) == \
        (initialfile, os.path.dirname(audio[0]))


def test_macos_skips_a_file_that_is_gone(monkeypatch, audio):
    """A name that does not resolve sends the macOS panel to the last-used folder."""
    os.remove(audio[0])
    assert _dialog_options(monkeypatch, 'aqua', audio[:1]) == ('', os.path.dirname(audio[0]))


@pytest.mark.parametrize('windowing_system', ['x11', 'aqua', 'win32'])
def test_nothing_loaded_yet(monkeypatch, windowing_system):
    assert _dialog_options(monkeypatch, windowing_system, []) == ('', '')
