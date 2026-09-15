"""Only the Windows dialog pre-selects the current audio files. Tk's own dialog,
used under X11, appends each confirmed file to -initialfile instead of replacing
it (tkfbox.tcl, ActivateEnt -> VerifyFileName: `lappend data(selectFile) $file`),
so the selection grew with every reopening (#340). The macOS panel has no name
field: it resolves -initialfile against -initialdir as the start location, and
the quoted list makes it open at the last-used folder instead of the audio folder.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

from noScribe import main


@pytest.mark.parametrize('windowing_system, expected', [
    ('x11', ''),
    ('aqua', ''),
    ('win32', '"interview_01.m4a" "interview_02.m4a"'),
], ids=['x11', 'aqua', 'win32'])
def test_preselection_only_in_the_windows_dialog(monkeypatch, windowing_system, expected):
    seen = {}

    def askopenfilename(**kwargs):
        seen.update(kwargs)
        return ''  # cancelled: the handler returns before touching the GUI

    def tk_call(*args):
        assert args == ('tk', 'windowingsystem')
        return windowing_system

    monkeypatch.setattr(main.tk.filedialog, 'askopenfilename', askopenfilename)
    app = SimpleNamespace(audio_files_list=('/audio/interview_01.m4a', '/audio/interview_02.m4a'),
                          tk=SimpleNamespace(call=tk_call))
    main.App.button_audio_file_event(app)
    assert seen['initialfile'] == expected
    assert seen['initialdir'] == '/audio'
