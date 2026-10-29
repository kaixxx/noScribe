"""Settings schema owned by the bundled Pyannote plugin."""

from dataclasses import asdict, dataclass

from ..settings import settings_values, validate_device


@dataclass
class PyannoteWorkerSettings:
    device: str = "auto"

    def __post_init__(self):
        validate_device(self.device, {"auto", "cpu", "cuda", "mps"})

    @classmethod
    def from_mapping(cls, value: dict, defaults=None):
        values = asdict(defaults) if defaults is not None else {}
        values.update(settings_values(value, {"device"}))
        return cls(**values)
