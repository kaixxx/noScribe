"""GPU errors fall back once per job without changing saved preferences."""
from types import SimpleNamespace

import pytest

import noScribe.main as m


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
    fallback_key = f'{component}_cpu_fallback'
    assert app_class._handle_cuda_fallback(None, component, RuntimeError('CUDA out of memory'), job) is (saved_value is not True)
    assert getattr(m, key) is (saved_value is True)
    assert getattr(job, fallback_key) is (saved_value is not True)
    other_component = 'whisper' if component == 'pyannote' else 'pyannote'
    assert getattr(job, f'{other_component}_cpu_fallback') is False
    assert config == original
    assert saved_configs == []
    # A CPU failure cannot cause an endless retry loop.
    assert app_class._handle_cuda_fallback(None, component, RuntimeError('CUDA error'), job) is False
    next_job = m.TranscriptionJob()
    assert getattr(next_job, fallback_key) is False
    assert app_class._handle_cuda_fallback(None, component, RuntimeError('CUDA error'), next_job) is (saved_value is not True)
    # Repeating the same job must also try the GPU again.
    job.set_running()
    assert getattr(job, fallback_key) is False
    assert app_class._handle_cuda_fallback(None, component, RuntimeError('CUDA error'), job) is (saved_value is not True)

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
@pytest.mark.parametrize('message', ['File not found', 'CUDA error (device_cpu)'])
def test_unrelated_or_cpu_errors_do_not_trigger_fallback(monkeypatch, component, message):
    monkeypatch.setattr(m, 'force_whisper_cpu', False)
    monkeypatch.setattr(m, 'force_pyannote_cpu', False)
    job = m.TranscriptionJob()
    assert m.App._handle_cuda_fallback(None, component, RuntimeError(message), job) is False
    assert job.whisper_cpu_fallback is False
    assert job.pyannote_cpu_fallback is False
    assert m.force_whisper_cpu is False
    assert m.force_pyannote_cpu is False
