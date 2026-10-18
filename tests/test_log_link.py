"""A log line with a link and a single tag given as a string.

`logn(t('rescue_saving', ...), 'error', link=...)` -- the notice that the
transcript could not be written under its name and was saved under another --
failed in the window: `'error' + ['hyper', 'hyper-0']` raised a TypeError, which
_append_log_text caught and wrote to the log file instead. So the only notice of
where a failed job's partial transcript went never appeared on screen.
"""
import queue
import threading

import pytest

pytest.importorskip("tkinter")

from noScribe.main import App


class _Textbox:
    def __init__(self):
        self.inserted = []

    def winfo_exists(self):
        return True

    def configure(self, **kwargs):
        pass

    def insert(self, index, txt, tags=None):
        self.inserted.append((txt, tags))

    def yview_moveto(self, fraction):
        pass


class _Hyperlinks:
    def add(self, action):
        return ["hyper", "hyper-0"]


@pytest.mark.parametrize("tags", ["error", ["error"]])
def test_link_with_a_tag_reaches_the_window(tags):
    app = object.__new__(App)
    app._headless = False
    app._shutting_down = False
    app._ui_thread_id = threading.get_ident()
    app._ui_tasks = queue.Queue()
    app.log_file = None
    app.log_len = 0
    app.log_textbox = _Textbox()
    app.hyperlink = _Hyperlinks()

    app.log("Saved as transcript (1).html\n", tags, link="file://transcript (1).html")

    assert app.log_textbox.inserted == [
        ("Saved as transcript (1).html\n", ["error", "hyper", "hyper-0"])
    ]
