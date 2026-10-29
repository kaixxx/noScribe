"""Lightweight error protocol shared by inference workers and their parent.

Keep this module free of torch and GUI imports: workers classify exceptions
before sending plain, serializable data across the process boundary.
"""
import re
import sys
import traceback


_GPU_ERROR = re.compile(
    r'\bcuda\s*(?:error|failed|failure|out of memory|driver|runtime|initialization)\b'
    r'|\bcudaError\w+\b'
    r'|\b(?:lib)?(?:cublas|cudnn|cufft|cudart|nvrtc)[\w.\-]*\b'
    r'|\b(?:device-side assert|invalid device function|no kernel image is available)\b'
    r'|\b(?:not compiled with CUDA|CUDA support is not enabled|GPU driver)\b',
    re.IGNORECASE,
)
_GPU_OOM = re.compile(
    r'\bcuda[^\n]*(?:out of memory|memory allocation)'
    r'|\bcudaErrorMemoryAllocation\b'
    r'|\b(?:cublas|cudnn)_status_alloc_failed\b', re.IGNORECASE,
)
_CPU_OOM = re.compile(r'\b(?:DefaultCPUAllocator|CPUAllocator)\b', re.IGNORECASE)


def classify_worker_error(error, device):
    """Use types first, then specific library diagnostics for generic errors."""
    if device in ('cpu', 'mps'):
        return 'other'
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (FileNotFoundError, PermissionError, IsADirectoryError, MemoryError)):
            return 'other'
        message = str(current)
        if _CPU_OOM.search(message):
            return 'other'
        # Inspect torch only if the worker already imported it successfully.
        torch = sys.modules.get('torch')
        cuda = getattr(torch, 'cuda', None)
        oom_type = getattr(cuda, 'OutOfMemoryError', None)
        accelerator_type = getattr(cuda, 'AcceleratorError', None)
        if device == 'cuda':
            if isinstance(oom_type, type) and isinstance(current, oom_type):
                return 'gpu_out_of_memory'
            if isinstance(accelerator_type, type) and isinstance(current, accelerator_type):
                return 'gpu_runtime'
        if _GPU_OOM.search(message):
            return 'gpu_out_of_memory'
        if _GPU_ERROR.search(message):
            return 'gpu_runtime'
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__)
    return 'other'


def error_result(error, device, stage):
    return {
        'type': 'result', 'ok': False,
        'error': str(error),
        'exception_type': f'{type(error).__module__}.{type(error).__qualname__}',
        'classification': classify_worker_error(error, device),
        'device': device, 'stage': stage,
        'trace': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


class WorkerState:
    """Report state before risky operations, including imports and model loading."""
    def __init__(self, queue, requested_device):
        self.queue = queue
        self.device = 'cpu' if requested_device == 'cpu' else None
        self.stage = 'imports'

    def update(self, stage, device=None):
        self.stage = stage
        if device is not None:
            self.device = device
        self.queue.put({'type': 'state', 'device': self.device, 'stage': self.stage})

    def failure(self, error):
        return error_result(error, self.device, self.stage)


class WorkerError(Exception):
    """A remote failure with its original diagnostics intact."""
    def __init__(self, result):
        self.message = result.get('error', 'Worker failed')
        self.exception_type = result.get('exception_type', 'Exception')
        self.classification = result.get('classification', 'other')
        self.device = result.get('device')
        self.stage = result.get('stage', 'unknown')
        self.trace = result.get('trace', '')
        self.exitcode = result.get('exitcode')
        super().__init__(
            f'{self.exception_type}: {self.message} '
            f'(stage={self.stage}, device={self.device or "unknown"})')

    @property
    def allows_cpu_retry(self):
        if self.classification == 'worker_crash':
            # A native CUDA failure can bypass Python exception handling. Only
            # retry if the worker explicitly reported selecting CUDA first.
            return self.device == 'cuda'
        return self.device not in ('cpu', 'mps') and self.classification in (
            'gpu_out_of_memory', 'gpu_runtime')

    @classmethod
    def unexpected_exit(cls, exitcode, state):
        return cls({
            **state, 'error': f'Worker exited unexpectedly (code {exitcode})',
            'exception_type': 'WorkerExit', 'classification': 'worker_crash',
            'exitcode': exitcode,
        })


class WorkerRetryError(Exception):
    """Keep both failures visible when the automatic CPU retry also fails."""
    def __init__(self, gpu_error, cpu_error):
        self.gpu_error = gpu_error
        self.cpu_error = cpu_error
        super().__init__(f'GPU attempt failed: {gpu_error}\nCPU retry failed: {cpu_error}')
