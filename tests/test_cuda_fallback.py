"""GPU errors fall back once per job without changing saved preferences."""
from types import SimpleNamespace

import pytest

import noScribe.main as m
from noScribe.worker_errors import WorkerError, WorkerRetryError, error_result


def gpu_error():
    return WorkerError(error_result(RuntimeError('CUDA out of memory'), 'cuda', 'transcription'))


@pytest.mark.parametrize('component', ['whisper', 'pyannote'])
@pytest.mark.parametrize('app_class', [m.App, m.HeadlessApp])
@pytest.mark.parametrize('saved_value', [None, False, True])
def test_fallback_and_ui_save_preserve_manual_overrides(monkeypatch, component, app_class, saved_value):
    key = f'force_{component}_cpu'
    config = {} if saved_value is None else {key: saved_value}
    original = config.copy()
    monkeypatch.setattr(m, 'config', config)
    monkeypatch.setattr(m, 'force_whisper_cpu', component == 'whisper' and saved_value is True)
    monkeypatch.setattr(m, 'force_pyannote_cpu', component == 'pyannote' and saved_value is True)
    def unexpected_prompt(**kwargs):
        pytest.fail('CPU fallback must not prompt, including in headless mode')
    monkeypatch.setattr(m.tk.messagebox, 'askyesno', unexpected_prompt)
    saved_configs = []
    monkeypatch.setattr(m, 'save_config', lambda: saved_configs.append(config.copy()))

    job = m.TranscriptionJob()
    app = SimpleNamespace(cancel=False, logn=lambda *args, **kwargs: None)
    fallback_key = f'{component}_cpu_fallback'
    original_error = gpu_error()
    assert app_class._handle_cuda_fallback(app, component, original_error, job) is (saved_value is not True)
    assert getattr(m, key) is (saved_value is True)
    assert getattr(job, fallback_key) is (saved_value is not True)
    other_component = 'whisper' if component == 'pyannote' else 'pyannote'
    assert getattr(job, f'{other_component}_cpu_fallback') is False
    assert config == original
    assert saved_configs == []
    # A CPU failure cannot cause an endless retry loop.
    cpu_error = WorkerError(error_result(RuntimeError('CPU failed'), 'cpu', 'transcription'))
    if saved_value is not True:
        with pytest.raises(WorkerRetryError) as failure:
            app_class._handle_cuda_fallback(app, component, cpu_error, job)
        assert failure.value.gpu_error is original_error
        assert failure.value.cpu_error is cpu_error
        assert 'CUDA out of memory' in str(failure.value)
        assert 'CPU failed' in str(failure.value)
    else:
        assert app_class._handle_cuda_fallback(app, component, cpu_error, job) is False
    next_job = m.TranscriptionJob()
    assert getattr(next_job, fallback_key) is False
    assert app_class._handle_cuda_fallback(app, component, gpu_error(), next_job) is (saved_value is not True)
    # Repeating the same job must also try the GPU again.
    job.set_running()
    assert getattr(job, fallback_key) is False
    assert app_class._handle_cuda_fallback(app, component, gpu_error(), job) is (saved_value is not True)

    control = SimpleNamespace(get=lambda: 'test')
    app = SimpleNamespace(**{name: control for name in (
        'option_menu_language', 'option_menu_speaker', 'option_menu_whisper_model',
        'option_menu_pause', 'check_box_overlapping', 'check_box_timestamps',
        'check_box_disfluencies',
    )})
    m.App.save_ui_state(app)
    assert len(saved_configs) == 1
    assert {k: v for k, v in saved_configs[0].items() if k.startswith('force_')} == original


@pytest.mark.parametrize('component', ['whisper', 'pyannote'])
@pytest.mark.parametrize('error', [
    RuntimeError('CUDA error in a parent callback'),
    WorkerError(error_result(FileNotFoundError('/cuda/input.wav'), 'cuda', 'audio_loading')),
    WorkerError(error_result(RuntimeError('CUDA error'), 'cpu', 'transcription')),
    WorkerError.unexpected_exit(-11, {'device': None, 'stage': 'imports'}),
])
def test_unrelated_or_cpu_errors_do_not_trigger_fallback(monkeypatch, component, error):
    monkeypatch.setattr(m, 'force_whisper_cpu', False)
    monkeypatch.setattr(m, 'force_pyannote_cpu', False)
    job = m.TranscriptionJob()
    assert m.App._handle_cuda_fallback(None, component, error, job) is False
    assert job.whisper_cpu_fallback is False
    assert job.pyannote_cpu_fallback is False
    assert m.force_whisper_cpu is False
    assert m.force_pyannote_cpu is False


@pytest.mark.parametrize('component', ['whisper', 'pyannote'])
def test_cancellation_during_cpu_retry_keeps_cancellation(monkeypatch, component):
    monkeypatch.setattr(m, 'force_whisper_cpu', False)
    monkeypatch.setattr(m, 'force_pyannote_cpu', False)
    app = SimpleNamespace(cancel=False, logn=lambda *args, **kwargs: None)
    job = m.TranscriptionJob()
    assert m.App._handle_cuda_fallback(app, component, gpu_error(), job)
    cancellation = RuntimeError(m.t('err_user_cancelation'))
    assert m.App._handle_cuda_fallback(app, component, cancellation, job) is False
    app.cancel = True
    assert m.App._handle_cuda_fallback(app, component, gpu_error(), job) is False


@pytest.mark.parametrize('component', ['whisper', 'pyannote'])
@pytest.mark.parametrize('device', ['cuda', 'cpu', 'mps', 'auto', None])
def test_crash_retries_only_after_cuda_was_selected(monkeypatch, component, device):
    monkeypatch.setattr(m, 'force_whisper_cpu', False)
    monkeypatch.setattr(m, 'force_pyannote_cpu', False)
    translations = []
    logs = []

    def translate(key, **kwargs):
        translations.append((key, kwargs))
        return key

    monkeypatch.setattr(m, 't', translate)
    app = SimpleNamespace(cancel=False, logn=lambda *args: logs.append(args))
    job = m.TranscriptionJob()
    crash = WorkerError.unexpected_exit(-11, {'device': device, 'stage': 'transcription'})
    assert m.App._handle_cuda_fallback(app, component, crash, job) is (device == 'cuda')
    assert getattr(job, f'{component}_cpu_fallback') is (device == 'cuda')
    if device == 'cuda':
        assert logs == [('gpu_worker_crash_retry', 'highlight')]
        details = next(kwargs for key, kwargs in translations if key == 'gpu_worker_crash_retry')
        assert details['exitcode'] == -11
        assert details['stage'] == 'transcription'
        assert details['option'] == f'force_{component}_cpu'
        cpu_crash = WorkerError.unexpected_exit(-11, {'device': 'cpu', 'stage': 'transcription'})
        with pytest.raises(WorkerRetryError) as failure:
            m.App._handle_cuda_fallback(app, component, cpu_crash, job)
        assert failure.value.gpu_error is crash
        assert failure.value.cpu_error is cpu_crash
        # A canceled job must never get a crash retry.
        app.cancel = True
        assert m.App._handle_cuda_fallback(app, component, crash, m.TranscriptionJob()) is False
    else:
        assert logs == []
