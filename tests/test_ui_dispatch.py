import queue
import threading

import pytest

pytest.importorskip("tkinter")

from noScribe import main
from noScribe.inference import InferenceWorkerError
from noScribe.main import App, _gui_error_message


def _app_stub():
    app = object.__new__(App)
    app._headless = False
    app._shutting_down = False
    app._ui_thread_id = threading.get_ident()
    app._ui_tasks = queue.Queue()
    return app


def test_background_ui_work_is_queued_for_main_thread():
    app = _app_stub()
    called_on = []

    worker = threading.Thread(
        target=lambda: app._dispatch_ui(
            lambda: called_on.append(threading.get_ident())
        )
    )
    worker.start()
    worker.join()

    assert called_on == []
    app._ui_tasks.get_nowait()()
    assert called_on == [app._ui_thread_id]


def test_main_thread_ui_work_runs_immediately():
    app = _app_stub()
    called_on = []

    app._dispatch_ui(lambda: called_on.append(threading.get_ident()))

    assert called_on == [app._ui_thread_id]
    assert app._ui_tasks.empty()


def test_background_progress_messages_replace_one_line_atomically():
    class FakeTextbox:
        def __init__(self):
            self.text = "embeddings: 50%"

        def winfo_exists(self):
            return True

        def configure(self, **_values):
            pass

        def get(self, _start, _end):
            return self.text

        def delete(self, _start, _end):
            self.text = ""

        def insert(self, _position, text, _tags):
            self.text += text

        def yview_moveto(self, _position):
            pass

    app = _app_stub()
    app.log_textbox = FakeTextbox()
    app.log_len = len(app.log_textbox.text)
    app.log_file = None

    worker = threading.Thread(target=lambda: (
        app.logr("embeddings: 75%", where="screen"),
        app.logr("embeddings: 100%", where="screen"),
    ))
    worker.start()
    worker.join()

    # Tk must not be read or written by the worker thread.
    assert app.log_textbox.text == "embeddings: 50%"
    while not app._ui_tasks.empty():
        app._ui_tasks.get_nowait()()

    assert app.log_textbox.text == "embeddings: 100%"
    assert app.log_len == len(app.log_textbox.text)


def test_gui_error_uses_short_message_from_wrapped_backend_error(monkeypatch):
    monkeypatch.setattr(
        main,
        "t",
        lambda message_id, **params: f"{message_id}: {params['name']}",
    )
    technical = InferenceWorkerError(
        "HTTPConnectionPool(host='server'): Max retries exceeded",
        user_message_id="err_remote_backend_connection",
        user_message_params={"name": "IfS-Server"},
    )
    try:
        raise RuntimeError("step failed") from technical
    except RuntimeError as wrapped:
        message = _gui_error_message(wrapped)

    assert "IfS-Server" in message
    assert "HTTPConnectionPool" not in message
