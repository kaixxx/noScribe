import queue
import threading

import pytest

pytest.importorskip("tkinter")

from noScribe.main import App


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
