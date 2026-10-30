"""The bundled local-voxtral plugin without its model: what it hands its worker,
what it refuses, and how main.py migrates saved choices, asks for job data and
leaves Voxtral out of the CPU fallback."""
from types import SimpleNamespace

import pytest

from noScribe.inference import DiarizationResult, DiarizationSegment, TranscriptionRequest
from noScribe.models import ModelRef
from noScribe.plugins.local_voxtral import LocalVoxtralPlugin
from noScribe.plugins.local_voxtral.settings import VoxtralWorkerSettings
from noScribe.worker_errors import WorkerError, error_result


class Host:
    """Records what the plugin asks the shared local backend to run."""

    def __init__(self):
        self.runs = []

    def run_transcription_worker(self, entrypoint, args, on_segment, **callbacks):
        self.runs.append((entrypoint.__name__, args, callbacks))
        return 'info'


def voxtral(tmp_path, monkeypatch, settings=None):
    from noScribe import voxtral_engine
    monkeypatch.setattr(voxtral_engine, 'resolve_model', lambda name, repo: f'repo/{name}')
    host = Host()
    return LocalVoxtralPlugin(host, corrections_dir=tmp_path, settings=settings), host


def test_voxtral_hands_its_worker_the_job(tmp_path, monkeypatch):
    """Language, the job data main.py adds (whether times are needed, the names),
    the diarization's turns in seconds, and its own settings -- 0 meaning "the
    engine decides", which the worker expects as None."""
    plugin, host = voxtral(tmp_path, monkeypatch, VoxtralWorkerSettings(chunk_sec=120, ram_reserve_gb=0))
    status = []
    diarization = DiarizationResult((DiarizationSegment(1500, 4200, 'SPEAKER_01'),))
    request = TranscriptionRequest(
        'a.wav', ModelRef('local-voxtral', 'voxtral-mini-8bit'), language='de', include_disfluencies=False,
        diarization=diarization,
        extensions={'noscribe': {'needs_timing': False, 'speaker_names': ['Mona', 'Lena']}})
    assert plugin.transcribe(request, on_segment=print,
                             on_status=lambda *message: status.append(message)) == 'info'
    (name, args, callbacks), = host.runs
    assert name == 'voxtral_proc_entrypoint'
    assert args == {
        'audio_path': 'a.wav', 'language_code': 'de', 'need_timestamps': False,
        'voxtral_repo': 'repo/voxtral-mini-8bit', 'chunk_sec': 120,
        'corrections_path': str(tmp_path / 'voxtral_corrections.yml'), 'speaker_names': ['Mona', 'Lena'],
        'ram_reserve_gb': None, 'speaker_turns': [[1.5, 4.2, 'SPEAKER_01']]}
    assert (tmp_path / 'voxtral_corrections.yml').exists()
    assert status == [('voxtral_path_short', {}, 'debug')]
    assert 'on_status' in callbacks


def test_voxtral_without_job_data_takes_the_safe_path(tmp_path, monkeypatch):
    """A request without main.py's job data (a server, a script) gets times, and
    disfluencies it cannot honour are said, not silently dropped."""
    plugin, host = voxtral(tmp_path, monkeypatch)
    status = []
    plugin.transcribe(TranscriptionRequest('a.wav', ModelRef('local-voxtral', 'voxtral-mini-8bit')),
                      on_segment=print, on_status=lambda *message: status.append(message[0]))
    (_, args, _), = host.runs
    assert args['need_timestamps'] is True and args['speaker_turns'] is None and args['speaker_names'] is None
    assert args['language_code'] is None
    assert status == ['voxtral_path_long', 'voxtral_no_disfluencies']


def test_voxtral_is_offered_on_apple_silicon_only(tmp_path, monkeypatch):
    from noScribe.plugins.local_voxtral import plugin as module
    plugin, _ = voxtral(tmp_path, monkeypatch)
    monkeypatch.setattr(module, '_apple_silicon', lambda: False)
    assert plugin.list_models() == []


def test_voxtral_settings_refuse_what_the_engine_cannot_use():
    assert VoxtralWorkerSettings.from_mapping({'ram_reserve_gb': 4}, VoxtralWorkerSettings(chunk_sec=90)) == \
        VoxtralWorkerSettings(chunk_sec=90, ram_reserve_gb=4)
    for bad in ({'chunk_sec': -1}, {'ram_reserve_gb': 'high'}, {'device': 'cuda'}):
        with pytest.raises(ValueError):
            VoxtralWorkerSettings.from_mapping(bad)


def test_the_registry_reads_voxtrals_settings_file(tmp_path):
    from noScribe.inference import LocalWorkerSettings
    from noScribe.plugins.factory import create_builtin_registry
    (tmp_path / 'plugin-settings').mkdir()
    (tmp_path / 'plugin-settings' / 'local-voxtral.yml').write_text('chunk_sec: 60\n')
    registry = create_builtin_registry({}, LocalWorkerSettings(), tmp_path / 'plugin-settings',
                                       voxtral_defaults=VoxtralWorkerSettings(ram_reserve_gb=6))
    assert registry.get('local-voxtral').settings == VoxtralWorkerSettings(chunk_sec=60, ram_reserve_gb=6)


def test_voxtral_does_not_retry_on_the_cpu(monkeypatch):
    """Voxtral runs on MLX; a "retry Whisper on the CPU" would only repeat the failure."""
    m = pytest.importorskip('noScribe.main')
    logged = []
    registry = SimpleNamespace(get=lambda backend_id: SimpleNamespace(settings=SimpleNamespace(device='auto')))
    app = SimpleNamespace(cancel=False, inference_backend=registry,
                          logn=lambda text='', *args, **kwargs: logged.append(text))
    job = m.TranscriptionJob()
    job.transcription_model = ModelRef('local-voxtral', 'voxtral-mini-8bit')
    error = WorkerError(error_result(RuntimeError('CUDA out of memory'), 'cuda', 'transcription'))
    assert m.App._handle_cuda_fallback(app, 'whisper', error, job) is False
    assert not job.whisper_cpu_fallback and logged == []


def test_job_data_tells_whether_the_output_needs_times():
    m = pytest.importorskip('noScribe.main')

    def needs(**options):
        options = {'speaker_detection': 'none', 'pause': 'none', 'timestamps': False, **options}
        job = m.create_transcription_job(transcript_file='x.txt', **options)
        return m._job_extension(job)['noscribe']['needs_timing']

    assert needs() is False
    assert needs(timestamps=True) is True
    assert needs(speaker_detection='auto') is True
    assert needs(pause='1sec+') is True
    job = m.create_transcription_job(transcript_file='x.html', speaker_names='Mona, Lena')
    assert m._job_extension(job) == {'noscribe': {'needs_timing': True, 'speaker_names': ['Mona', 'Lena']}}


def test_the_shared_worker_loop_runs_voxtrals_arguments(tmp_path, monkeypatch):
    """The plugins run their workers through LocalInferenceBackend's own loop,
    which was written for Whisper's arguments: Voxtral has no device setting, and
    a loop that read worker_args['device'] failed every Voxtral job before the
    worker started. Here the real loop runs the real arguments, with a worker
    that answers in a thread instead of a spawned process."""
    import queue
    import threading
    from noScribe import inference, voxtral_mp_worker
    from noScribe.inference import LocalInferenceBackend

    class Process:
        def __init__(self, target, args):
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.exitcode = 0

        def start(self):
            self.thread.start()

        def is_alive(self):
            return self.thread.is_alive()

        def join(self, timeout=None):
            self.thread.join(timeout)

        def terminate(self):
            pass

        def close(self):
            pass

    class Queue(queue.Queue):
        def close(self):
            pass

        def join_thread(self):
            pass

    seen = []

    def worker(args, q):
        seen.append(args)
        q.put({"type": "progress", "pct": 50})
        q.put({"type": "segment", "segment": {"start": 0.0, "end": 1.0, "text": " Hallo", "words": []}})
        q.put({"type": "result", "ok": True, "info": {"duration": 1.0, "language": "de"}})

    monkeypatch.setattr(inference.mp, "get_context", lambda method: SimpleNamespace(Queue=Queue, Process=Process))
    monkeypatch.setattr(voxtral_mp_worker, "voxtral_proc_entrypoint", worker)
    from noScribe import voxtral_engine
    monkeypatch.setattr(voxtral_engine, "resolve_model", lambda name, repo: f"repo/{name}")
    plugin = LocalVoxtralPlugin(LocalInferenceBackend(), corrections_dir=tmp_path)
    segments, progress = [], []
    info = plugin.transcribe(TranscriptionRequest("a.wav", ModelRef("local-voxtral", "voxtral-mini-8bit")),
                             on_segment=segments.append, on_progress=lambda pct, detail: progress.append(pct))
    assert "device" not in seen[0]
    assert [segment.text for segment in segments] == [" Hallo"] and progress == [50.0]
    assert info.language == "de" and info.duration == 1.0


# --- the model menu and saved choices -----------------------------------------

def _registry_with_voxtral(names=("voxtral-mini-8bit", "voxtral-small-4bit")):
    """The real registry, with a Voxtral plugin that offers `names` whatever
    the machine (the real one offers its builds on Apple Silicon only)."""
    from noScribe.inference import LocalWorkerSettings
    from noScribe.models import ModelDescriptor
    from noScribe.plugins.factory import create_builtin_registry

    registry = create_builtin_registry({"precise": "models/precise"}, LocalWorkerSettings())
    plugin = registry.get("local-voxtral")
    plugin.list_models = lambda: [ModelDescriptor(
        ref=ModelRef("local-voxtral", name), display_name=name, engine="voxtral",
        capabilities=plugin.manifest.capabilities) for name in names]
    return registry


def test_model_ram_hint_shows_in_the_menu_but_not_in_the_header():
    """The menu shows the RAM a Voxtral model needs; the selection still finds
    the model behind the decorated entry, and the transcript header and the
    queue name it without the hint."""
    m = pytest.importorskip("noScribe.main")
    app = SimpleNamespace(inference_backend=_registry_with_voxtral())
    m._update_transcription_model_catalog(app)
    label = app.transcription_model_labels["local-voxtral:voxtral-mini-8bit"]
    assert label.startswith("voxtral-mini-8bit") and "GB RAM" in label
    assert str(app.transcription_model_options[label].ref) == "local-voxtral:voxtral-mini-8bit"
    assert app.transcription_model_names["local-voxtral:voxtral-mini-8bit"] == "voxtral-mini-8bit"
    # Whisper models are shown unchanged.
    assert app.transcription_model_labels["local-whisper:precise"] == "precise"


def test_a_bare_model_name_finds_the_local_plugin_that_offers_it():
    """`--model voxtral-mini-8bit` works as `--model precise` does; a name
    Whisper offers stays Whisper's, even where another plugin offers it too."""
    m = pytest.importorskip("noScribe.main")
    registry = _registry_with_voxtral(("voxtral-mini-8bit", "precise"))
    assert str(m._model_ref("voxtral-mini-8bit", registry)) == "local-voxtral:voxtral-mini-8bit"
    assert str(m._model_ref("precise", registry)) == "local-whisper:precise"
    assert str(m._model_ref("unknown", registry)) == "local-whisper:unknown"
    assert str(m._model_ref("local-voxtral:voxtral-mini-8bit")) == "local-voxtral:voxtral-mini-8bit"


def test_voxtral_offers_its_builds_where_it_can_run_them(tmp_path, monkeypatch):
    from noScribe import voxtral_engine
    from noScribe.plugins.local_voxtral import plugin as module
    plugin, _ = voxtral(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_apple_silicon", lambda: True)
    monkeypatch.setattr(voxtral_engine, "is_available", lambda: True)
    models = plugin.list_models()
    assert {model.ref.model_id for model in models} == set(voxtral_engine.VOXTRAL_MODELS)
    assert all(model.ref.backend_id == "local-voxtral" and model.engine == "voxtral"
               and "transcription" in model.capabilities for model in models)
    monkeypatch.setattr(voxtral_engine, "is_available", lambda: False)
    assert plugin.list_models() == []


def test_an_unknown_model_is_refused_as_a_value_error(tmp_path, monkeypatch):
    plugin, host = voxtral(tmp_path, monkeypatch)
    plugin._repo = LocalVoxtralPlugin._repo.__get__(plugin)
    with pytest.raises(ValueError):
        plugin._repo("voxtral-unknown")


# --- the worker's failures and the progress bar -------------------------------

def test_a_failing_worker_reports_where_it_failed(monkeypatch):
    """Through worker_errors.error_result, as every local worker does: the user
    sees the stage and the device, not "stage=unknown, device=unknown"."""
    from noScribe import voxtral_engine, voxtral_mp_worker
    from noScribe.plugins.protocol import validate_worker_event

    def fail(**kwargs):
        raise MemoryError("the pass did not fit")

    monkeypatch.setattr(voxtral_engine, "transcribe", fail)
    sent = []
    voxtral_mp_worker.voxtral_proc_entrypoint({"audio_path": "a.wav"}, SimpleNamespace(put=sent.append))
    result = sent[-1]
    validate_worker_event(result)
    assert result["ok"] is False and result["stage"] == "transcription" and result["device"] == "mps"
    assert result["exception_type"] == "builtins.MemoryError"
    error = WorkerError(result)
    assert "stage=transcription" in str(error) and not error.allows_cpu_retry


def test_the_bar_follows_the_segments_again_after_a_voxtral_job(tmp_path, monkeypatch):
    """Voxtral reports progress of its own, and on_segment then leaves the bar
    alone. A Whisper job on a server after it, which reports none, must move the
    bar by its segments again."""
    m = pytest.importorskip("noScribe.main")
    soundfile = pytest.importorskip("soundfile")
    import numpy as np
    from noScribe.inference import InferenceWorkflowResult, TranscriptionInfo, TranscriptionSegment
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    monkeypatch.setattr(m, "get_speech_timestamps", lambda *args: [])
    app = m.HeadlessApp()
    for name in ("log", "logn", "logr", "update_queue_table", "update_queue_controls"):
        monkeypatch.setattr(app, name, lambda *args, **kwargs: None)
    bar = []
    monkeypatch.setattr(app, "set_progress", lambda step, value, *args: bar.append((step, value)))

    def run_workflow(request, **callbacks):
        callbacks["on_diarization_result"](DiarizationResult((DiarizationSegment(0, 3000, "SPEAKER_00"),)))
        callbacks["on_transcription_segment"](TranscriptionSegment(0.2, 1.5, " Hallo"))
        return InferenceWorkflowResult(transcription_info=TranscriptionInfo())

    monkeypatch.setattr(app.inference_backend, "supports_workflow", lambda *args: True)
    monkeypatch.setattr(app.inference_backend, "run_workflow", run_workflow)
    monkeypatch.setattr(m, "model_uses_remote_backend", lambda *args: True)
    app._engine_progress = True  # as a Voxtral job before it leaves it
    audio = tmp_path / "a.wav"
    soundfile.write(audio, np.zeros(16000 * 3, dtype=np.int16), 16000)
    job = m.create_transcription_job(audio_file=str(audio), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection="auto", cli_mode=True)
    job.transcription_model = ModelRef("lab", "local-whisper/precise")
    job.diarization_model = ModelRef("lab", "local-pyannote/default")
    app.queue.add_job(job)
    app.transcription_worker()
    assert job.status == m.JobStatus.FINISHED, job.error_message
    assert (3, 50) in bar  # 1.5 s of 3 s
