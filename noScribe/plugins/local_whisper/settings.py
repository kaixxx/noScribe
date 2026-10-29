"""Settings schema owned by the bundled Whisper plugin."""

from dataclasses import asdict, dataclass

from ..settings import settings_values, validate_device


@dataclass
class WhisperWorkerSettings:
    device: str = "auto"
    cpu_threads: int = 4
    vad_threshold: float = 0.5

    def __post_init__(self):
        validate_device(self.device, {"auto", "cpu", "cuda"})
        if type(self.cpu_threads) is not int or self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive integer.")
        if (type(self.vad_threshold) not in (float, int)
                or not 0 <= self.vad_threshold <= 1):
            raise ValueError("vad_threshold must be between 0 and 1.")

    @classmethod
    def from_mapping(cls, value: dict, defaults=None):
        values = asdict(defaults) if defaults is not None else {}
        values.update(settings_values(value, {"device", "cpu_threads", "vad_threshold"}))
        return cls(**values)
