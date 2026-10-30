"""The bundled local-nemotron plugin without its model: what it hands its worker,
what it refuses, how its result travels, and how main.py chooses the diarization
engine and falls back to the CPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from noScribe import nemotron_mp_worker as nw
from noScribe.inference import DiarizationRequest, DiarizationResult, LocalWorkerSettings
from noScribe.models import ModelDescriptor, ModelRef
from noScribe.plugins.factory import create_builtin_registry
from noScribe.plugins.local_nemotron import LocalNemotronPlugin
from noScribe.plugins.local_nemotron.settings import NemotronWorkerSettings
from noScribe.plugins.protocol import validate_extensions
from noScribe.worker_errors import WorkerError, error_result


class Host:
    """Records what the plugin asks the shared local backend to run."""

    def __init__(self):
        self.runs = []

    def run_diarization_worker(self, entrypoint, args, **callbacks):
        self.runs.append((entrypoint.__name__, args, callbacks))
        return DiarizationResult()


def test_nemotron_refuses_a_fixed_number_of_speakers():
    """It cannot be held to one; main.py keeps such jobs with Pyannote, and the
    plugin says so rather than ignoring the number."""
    host = Host()
    plugin = LocalNemotronPlugin(host)
    with pytest.raises(ValueError):
        plugin.diarize(DiarizationRequest('a.wav', ModelRef('local-nemotron', 'default'), num_speakers=2))
    assert host.runs == []


def test_nemotron_runs_on_its_device_or_the_override():
    host = Host()
    plugin = LocalNemotronPlugin(host, settings=NemotronWorkerSettings(device='mps'))
    request = DiarizationRequest('a.wav', ModelRef('local-nemotron', 'default'))
    plugin.diarize(request)
    plugin.diarize(request, device_override='cpu')
    assert [(name, args['device']) for name, args, _ in host.runs] == [
        ('nemotron_proc_entrypoint', 'mps'), ('nemotron_proc_entrypoint', 'cpu')]


def test_nemotron_force_cpu_is_remembered(tmp_path):
    from noScribe.plugins.settings import PluginSettingsStore
    store = PluginSettingsStore(tmp_path)
    LocalNemotronPlugin(Host(), settings_store=store).force_cpu()
    assert store.load('local-nemotron') == {'device': 'cpu'}


def test_nemotrons_probabilities_travel_as_data_and_come_back_exactly():
    """A server cannot hand the desktop a file path, so the probabilities travel
    in the result: JSON-compatible, as extensions must be, and bit for bit the
    float16 rows the worker computed."""
    rows = np.random.default_rng(0).random((997, 8)).astype(np.float16)
    data = nw.encode_probabilities(rows, [nw.label(k) for k in range(8)])
    validate_extensions({nw.EXTENSION_KEY: {'probabilities': data}})
    back = nw.decode_probabilities(json.loads(json.dumps(data)))
    assert back.dtype == np.float16 and back.shape == rows.shape
    assert back.tobytes() == rows.tobytes()
    assert data['columns'][0] == 'SPEAKER_00' and data['frame_s'] == nw.FRAME_S


def test_the_registry_reads_nemotrons_settings_file(tmp_path):
    (tmp_path / 'plugin-settings').mkdir()
    registry = create_builtin_registry({}, LocalWorkerSettings(), tmp_path / 'plugin-settings',
                                       nemotron_defaults=NemotronWorkerSettings(device='mps'))
    assert registry.get('local-nemotron').settings == NemotronWorkerSettings(device='mps')
    (tmp_path / 'plugin-settings' / 'local-nemotron.yml').write_text('device: cpu\n')
    registry = create_builtin_registry({}, LocalWorkerSettings(), tmp_path / 'plugin-settings',
                                       nemotron_defaults=NemotronWorkerSettings(device='mps'))
    assert registry.get('local-nemotron').settings == NemotronWorkerSettings(device='cpu')


# --- main.py's choice of the diarization engine ---------------------------

class Registry:
    def __init__(self, *refs, ready=True):
        self.models = [ModelDescriptor(ref=ModelRef.parse(ref), display_name=ref, engine='x',
                                       capabilities=frozenset({'diarization'})) for ref in refs]
        self.nemotron = SimpleNamespace(ready=lambda: ready)

    def list_models(self, capability=None):
        return self.models

    def get(self, backend_id):
        assert backend_id == 'local-nemotron'
        return self.nemotron


@pytest.mark.parametrize('setting, ready, speakers, chosen', [
    ('auto', True, None, 'local-nemotron:default'),
    ('auto', False, None, 'local-pyannote:default'),    # never starts a download
    ('auto', True, 2, 'local-pyannote:default'),        # a fixed number stays with Pyannote
    ('nemotron', False, None, 'local-nemotron:default'),  # downloads on first use
    ('nemotron', True, 3, 'local-pyannote:default'),
    ('pyannote', True, None, 'local-pyannote:default'),
])
def test_the_diarization_engine_follows_the_setting(monkeypatch, setting, ready, speakers, chosen):
    m = pytest.importorskip('noScribe.main')
    monkeypatch.setattr(m, 'config', {'diarization_engine': setting})
    registry = Registry('local-pyannote:default', 'local-nemotron:default', ready=ready)
    assert str(m.select_diarization_model(
        registry, ModelRef('local-whisper', 'precise'), speakers)) == chosen


def test_a_server_diarizes_with_its_own_engine(monkeypatch):
    m = pytest.importorskip('noScribe.main')
    monkeypatch.setattr(m, 'config', {})
    registry = Registry('local-pyannote:default', 'local-nemotron:default', 'server:local-pyannote/default')
    assert str(m.select_diarization_model(registry, ModelRef('server', 'local-whisper/precise'))) == \
        'server:local-pyannote/default'


def test_without_nemotron_installed_auto_stays_with_pyannote(monkeypatch):
    m = pytest.importorskip('noScribe.main')
    monkeypatch.setattr(m, 'config', {'diarization_engine': 'nemotron'})
    assert str(m.select_diarization_model(Registry('local-pyannote:default'),
                                          ModelRef('local-whisper', 'precise'))) == 'local-pyannote:default'


# --- what the job says about the engine ------------------------------------

def test_the_header_names_the_engine_behind_the_speakers():
    m = pytest.importorskip('noScribe.main')
    job = m.create_transcription_job(transcript_file='x.html', speaker_detection='auto')
    assert '(Nemotron)' in m.format_transcript_options(job, 'precise', 'Nemotron')
    job = m.create_transcription_job(transcript_file='x.html', speaker_detection='none')
    assert '(Nemotron)' not in m.format_transcript_options(job, 'precise', 'Nemotron')


@pytest.mark.parametrize('engine, speakers, said', [
    ('nemotron', 8, True), ('nemotron', 7, False), ('pyannote', 9, False)])
def test_all_eight_channels_in_use_are_said(monkeypatch, engine, speakers, said):
    """Nemotron folds a ninth voice into the others without knowing it."""
    m = pytest.importorskip('noScribe.main')
    from noScribe.inference import DiarizationSegment
    monkeypatch.setattr(m, 't', lambda key, **kwargs: key)
    logged = []
    app = SimpleNamespace(logn=lambda text='', *args, **kwargs: logged.append(text),
                          _model_engine=lambda model, default: engine)
    job = SimpleNamespace(diarization_model=ModelRef('local-' + engine, 'default'))
    segments = [DiarizationSegment(k * 1000, k * 1000 + 500, f'SPEAKER_{k:02d}') for k in range(speakers)]
    m.App._say_if_all_channels_used(app, job, segments)
    assert (logged == ['nemotron_all_speakers']) is said


# --- the CPU fallback covers Nemotron -------------------------------------

def test_nemotron_retries_on_the_cpu_and_says_so(monkeypatch):
    m = pytest.importorskip('noScribe.main')
    monkeypatch.setattr(m, 't', lambda key, **kwargs: key)
    logged = []
    registry = SimpleNamespace(get=lambda backend_id: SimpleNamespace(settings=SimpleNamespace(device='auto')))
    app = SimpleNamespace(cancel=False, inference_backend=registry,
                          logn=lambda text='', *args, **kwargs: logged.append(text))
    job = m.TranscriptionJob()
    job.diarization_model = ModelRef('local-nemotron', 'default')
    error = WorkerError(error_result(RuntimeError('CUDA out of memory'), 'cuda', 'diarization'))
    assert m.App._handle_cuda_fallback(app, 'pyannote', error, job) is True
    assert job.pyannote_cpu_fallback and logged == ['nemotron_cuda_retry']
