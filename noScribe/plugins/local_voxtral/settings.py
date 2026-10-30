"""Settings schema owned by the bundled Voxtral plugin."""

from dataclasses import asdict, dataclass

from ..settings import settings_values


@dataclass
class VoxtralWorkerSettings:
    # Seconds per pass; 0 lets the engine size each pass from the free RAM.
    chunk_sec: float = 0
    # GB the engine leaves free for everything else; 0 keeps its own default.
    ram_reserve_gb: float = 0

    def __post_init__(self):
        for name in ("chunk_sec", "ram_reserve_gb"):
            value = getattr(self, name)
            if type(value) not in (float, int) or value < 0:
                raise ValueError(f"{name} must be a number of at least 0.")

    @classmethod
    def from_mapping(cls, value: dict, defaults=None):
        values = asdict(defaults) if defaults is not None else {}
        values.update(settings_values(value, {"chunk_sec", "ram_reserve_gb"}))
        return cls(**values)
