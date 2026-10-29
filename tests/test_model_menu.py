from types import SimpleNamespace

import pytest

import noScribe.main as main
from noScribe.models import ModelDescriptor, ModelRef
from noScribe.plugins import factory
from noScribe.plugins.factory import register_remote_profiles
from noScribe.plugins.manifest import ExecutionType
from noScribe.plugins.registry import BackendRegistry
from noScribe.plugins.remote_profiles import RemoteBackendProfile


def _model(backend, model_id):
    return ModelDescriptor(ModelRef(backend, model_id), model_id, "test",
                           frozenset({"transcription"}))


class Menu:
    def __init__(self, value):
        self.value = value
        self.values = []

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def configure(self, *, values):
        self.values = values


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(main, "t", lambda key, **kwargs: key)
    models = [_model("local-whisper", "fast"), _model("local-whisper", "precise")]
    local = SimpleNamespace(
        manifest=SimpleNamespace(id="local-whisper", name="Whisper",
                                 execution=SimpleNamespace(type=ExecutionType.BUILTIN)),
        list_models=lambda: models,
    )
    ui_tasks = []
    app = SimpleNamespace(
        inference_backend=BackendRegistry([local]), remote_backend_profiles=(),
        option_menu_whisper_model=Menu("fast"),
        _remote_model_refresh_in_progress=False, _remote_model_refresh_pending=False,
        _remote_models_refreshed_at=float("-inf"), _last_remote_refresh_errors=(),
        _worker_threads=[], _dispatch_ui=ui_tasks.append,
        logn=lambda *args, **kwargs: None,
        ui_tasks=ui_tasks,
    )
    app.refresh_remote_models = lambda **kwargs: main.App.refresh_remote_models(app, **kwargs)
    main._update_transcription_model_catalog(app)
    return app


def _finish_refresh(app):
    assert app._worker_threads
    app._worker_threads[0].join(timeout=3)
    assert not app._worker_threads[0].is_alive()
    app.ui_tasks.pop(0)()
    assert not app._remote_model_refresh_in_progress


def test_opening_menu_retries_server_that_was_offline_at_startup(app, monkeypatch):
    state = {"online": False, "attempts": 0}

    class RemotePlugin:
        def __init__(self, profile):
            state["attempts"] += 1
            if not state["online"]:
                raise ConnectionError("offline")
            self.profile = profile
            self.manifest = SimpleNamespace(
                id=profile.id, name=profile.name,
                execution=SimpleNamespace(type=ExecutionType.REMOTE),
            )

        def list_models(self):
            return [_model(self.profile.id, "remote-model")]

        def refresh_models(self):
            state["attempts"] += 1

    monkeypatch.setattr(factory, "RemoteHttpPlugin", RemotePlugin)
    app.remote_backend_profiles = (RemoteBackendProfile(
        id="server", name="Server", driver="noscribe-http-v1",
        url="http://127.0.0.1:8766", api_key="test",
    ),)
    assert register_remote_profiles(app.inference_backend, app.remote_backend_profiles)
    assert state["attempts"] == 1

    # First opening still fails. A second opening immediately retries, rather
    # than waiting out the refresh interval after that failed request.
    main._prepare_model_menu(app, app.option_menu_whisper_model)
    _finish_refresh(app)
    assert state["attempts"] == 2
    state["online"] = True
    main._prepare_model_menu(app, app.option_menu_whisper_model)
    _finish_refresh(app)

    assert state["attempts"] == 3
    assert "remote-model (Server)" in app.option_menu_whisper_model.values
    assert app.option_menu_whisper_model.get() == "fast"


def test_missing_last_used_model_prefers_local_precise(app):
    assert main._model_selection_label(app, "server:unavailable") == "precise"
    assert main._model_selection_label(app, "local-whisper:fast") == "fast"


def test_refresh_uses_precise_if_selected_model_disappears(app):
    unavailable = _model("server", "remote-model")
    app.transcription_model_options["remote-model (Server)"] = unavailable
    app.transcription_model_labels[str(unavailable.ref)] = "remote-model (Server)"
    app.option_menu_whisper_model.set("remote-model (Server)")
    main._prepare_model_menu(app, app.option_menu_whisper_model)
    _finish_refresh(app)
    assert app.option_menu_whisper_model.get() == "precise"


def test_refresh_preserves_selection_changed_while_request_was_running(app):
    main._prepare_model_menu(app, app.option_menu_whisper_model)
    app.option_menu_whisper_model.set("precise")
    _finish_refresh(app)
    assert app.option_menu_whisper_model.get() == "precise"


def test_no_precise_model_uses_first_available_or_none(app):
    del app.transcription_model_options["precise"]
    del app.transcription_model_labels["local-whisper:precise"]
    assert main._model_selection_label(app, "server:missing") == "fast"
    app.transcription_model_options.clear()
    app.transcription_model_labels.clear()
    assert main._model_selection_label(app, None) is None
