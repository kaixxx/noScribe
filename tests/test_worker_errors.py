"""Error classification and transport without requiring a working GPU."""
import builtins
from queue import Empty, Queue
import sys
from types import SimpleNamespace

import pytest

from noScribe.worker_errors import WorkerError, classify_worker_error, error_result


@pytest.mark.parametrize('error, device, classification', [
    (RuntimeError('CUDA failed with error out of memory'), 'cuda', 'gpu_out_of_memory'),
    (RuntimeError('CUDA error: invalid device ordinal'), 'cuda', 'gpu_runtime'),
    (RuntimeError('CUDA driver version is insufficient for CUDA runtime version'), None, 'gpu_runtime'),
    (ImportError('Library libcudnn.so.9 is not found'), None, 'gpu_runtime'),
    (RuntimeError('cublas64_12.dll is not found'), None, 'gpu_runtime'),
    (RuntimeError('CUBLAS_STATUS_ALLOC_FAILED'), 'cuda', 'gpu_out_of_memory'),
    (RuntimeError('no kernel image is available for execution on the device'), 'cuda', 'gpu_runtime'),
    (RuntimeError('DefaultCPUAllocator: not enough memory'), 'cuda', 'other'),
    (MemoryError('CUDA out of memory'), 'cuda', 'other'),
    (RuntimeError('CUDA out of memory'), 'cpu', 'other'),
    (RuntimeError('CUDA error'), 'mps', 'other'),
    (FileNotFoundError('/models/cuda error/model.bin'), 'cuda', 'other'),
    (PermissionError('/models/libcudnn.so'), None, 'other'),
    (RuntimeError('Could not decode /audio/cuda/interview.wav'), 'cuda', 'other'),
    (RuntimeError('out of memory'), 'cuda', 'other'),
    (ValueError('invalid waveform shape'), 'cuda', 'other'),
])
def test_classification(error, device, classification):
    assert classify_worker_error(error, device) == classification


def test_typed_pytorch_errors_need_no_diagnostic_keywords(monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    class AcceleratorError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=OutOfMemoryError, AcceleratorError=AcceleratorError)))
    assert classify_worker_error(OutOfMemoryError('allocation failed'), 'cuda') == 'gpu_out_of_memory'
    assert classify_worker_error(OutOfMemoryError('allocation failed'), 'cpu') == 'other'
    assert classify_worker_error(AcceleratorError('launch failed'), 'cuda') == 'gpu_runtime'


def test_chained_error_retains_trace_and_classification():
    try:
        try:
            raise RuntimeError('CUDA error: launch failed')
        except RuntimeError as cause:
            raise RuntimeError('Model initialization failed') from cause
    except RuntimeError as error:
        result = error_result(error, 'cuda', 'model_loading')
    transported = WorkerError(result)
    assert transported.allows_cpu_retry
    assert transported.stage == 'model_loading'
    assert transported.exception_type == 'builtins.RuntimeError'
    assert 'CUDA error: launch failed' in transported.trace
    assert 'Model initialization failed' in str(transported)


@pytest.mark.parametrize('worker_module, failing_import', [
    ('whisper_mp_worker', 'faster_whisper'), ('pyannote_mp_worker', 'torch'),
])
def test_workers_report_import_errors_before_device_selection(monkeypatch, worker_module, failing_import):
    from noScribe import whisper_mp_worker, pyannote_mp_worker
    module = {'whisper_mp_worker': whisper_mp_worker, 'pyannote_mp_worker': pyannote_mp_worker}[worker_module]
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == failing_import:
            raise ImportError('Could not load libcudnn.so.9')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    messages = Queue()
    entrypoint = getattr(module, 'whisper_proc_entrypoint' if worker_module == 'whisper_mp_worker' else 'pyannote_proc_entrypoint')
    entrypoint({'device': 'auto'}, messages)
    result = list(messages.queue)[-1]
    assert result['ok'] is False
    assert result['stage'] == 'imports'
    assert result['device'] is None
    assert result['exception_type'] == 'builtins.ImportError'
    assert result['classification'] == 'gpu_runtime'
    assert 'libcudnn.so.9' in result['trace']


@pytest.mark.parametrize('failed_stage', ['model_loading', 'transcription'])
def test_whisper_reports_selected_device_and_inference_failures(monkeypatch, tmp_path, failed_stage):
    import torch
    from noScribe.whisper_mp_worker import whisper_proc_entrypoint

    monkeypatch.setattr('noScribe.whisper_mp_worker.platform.system', lambda: 'Windows')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)

    class Model:
        model = SimpleNamespace(device='cuda', is_multilingual=True)

        def __init__(self, *args, **kwargs):
            assert kwargs['device'] == 'cuda'
            if failed_stage == 'model_loading':
                raise RuntimeError('CUDA out of memory')

        def transcribe(self, *args, **kwargs):
            def segments():
                yield SimpleNamespace(start=0, end=1, text='First segment')
                raise RuntimeError('CUDA error: launch failed')
            return segments(), SimpleNamespace(duration=2)

    monkeypatch.setattr('faster_whisper.WhisperModel', Model)
    audio = tmp_path / 'audio.wav'
    audio.touch()
    messages = Queue()
    whisper_proc_entrypoint({
        'device': 'auto', 'model_path': str(tmp_path),
        'audio_path': str(audio), 'language': 'de',
    }, messages)
    result = list(messages.queue)[-1]
    assert result['ok'] is False
    assert result['device'] == 'cuda'
    assert result['stage'] == failed_stage
    assert WorkerError(result).allows_cpu_retry
    if failed_stage == 'transcription':
        assert any(msg['type'] == 'segment' for msg in messages.queue)


@pytest.mark.parametrize('component', ['whisper', 'pyannote'])
@pytest.mark.parametrize('crashed', [False, True])
def test_parent_preserves_failure_details_and_last_state(monkeypatch, component, crashed):
    import noScribe.inference as inference
    failure = error_result(RuntimeError('CUDA out of memory'), 'cuda', 'model_loading')
    messages = [{'type': 'state', 'device': 'cuda', 'stage': 'model_loading'}]
    if not crashed:
        messages.append(failure)

    class FakeQueue:
        def get(self, **kwargs):
            if messages:
                return messages.pop(0)
            raise Empty

        def close(self):
            pass

        def join_thread(self):
            pass

    process = SimpleNamespace(start=lambda: None, is_alive=lambda: False,
                              exitcode=-11, join=lambda **kw: None, close=lambda: None)
    context = SimpleNamespace(Queue=FakeQueue, Process=lambda **kw: process)
    monkeypatch.setattr(inference.mp, 'get_context', lambda *args: context)
    backend = inference.LocalInferenceBackend(whisper_models={'precise': 'model'})
    with pytest.raises(WorkerError) as caught:
        if component == 'whisper':
            backend.transcribe(inference.TranscriptionRequest(
                'test.wav', inference.ModelRef('local-whisper', 'precise')),
                on_segment=lambda segment: None)
        else:
            backend.diarize(inference.DiarizationRequest('test.wav'))
    error = caught.value
    assert error.device == 'cuda'
    assert error.stage == 'model_loading'
    assert error.allows_cpu_retry is True
    if crashed:
        assert error.classification == 'worker_crash'
        assert error.exitcode == -11
        assert 'code -11' in str(error)
    else:
        assert error.trace == failure['trace']
        assert error.classification == 'gpu_out_of_memory'
    assert backend._process is None
    assert backend._queue is None
