import dataclasses
from typing import Optional
import importlib.resources as impres
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DIR_PACKAGE_MODELS = "models"


@dataclasses.dataclass
class WhisperModel:
    """
    Represents a whisper model or more specifically a model that can be used
    for transcriptions.

    `engine` selects the backend: "whisper" (faster-whisper, the default) or
    "voxtral" (Mistral Voxtral via mlx-voxtral). For the Voxtral engine `repo`
    holds the model repository/path and `path` is only a display placeholder
    that no caller may read.
    """

    name: str
    path: Path
    engine: str = "whisper"
    repo: Optional[str] = None


class WhisperModelManager:
    """
    Handles whisper models. Models can either be in the package directory or in
    given additional paths.

    Currently, it supports only to get a list of available models. In the
    future, it can be used to
    """

    def __init__(self, path_user_dir: Path | None = None):
        self.models: dict = {}
        self.path_user_dir = path_user_dir

        # Collect models in project directory.
        self._collect_whisper_models(impres.files(DIR_PACKAGE_MODELS))

        # Collect models in user directory.
        self._collect_whisper_models(path_user_dir)

    def get_installed_models(self):
        return self.models

    def _collect_whisper_models(self, curpath: Path):
        if not curpath.is_dir():
            logger.warning("Given model path is not a directory: %s.", curpath)
            return

        for entry in curpath.iterdir():
            if not entry.is_dir():
                continue

            if entry.name in self.models:
                logger.warning(
                    "Found duplicate model name: %s (%s).",
                    entry.name,
                    entry.absolute(),
                )
                continue

            # faster-whisper models have a `model.bin`. A directory without one
            # is either a different model format -- e.g. an MLX/Voxtral build,
            # which uses safetensors and (for the shipped builds) is registered
            # separately by its own engine -- or an incomplete/broken
            # faster-whisper download. A safetensors dir is not a faster-whisper
            # model, so we don't warn (that was noise for the expected Voxtral
            # case); but we still log it at debug so a user-supplied safetensors
            # model that never shows up in the picker is diagnosable. A dir with
            # neither `model.bin` nor safetensors is a broken download -- warn.
            if not (entry / "model.bin").exists():
                if any(entry.glob("*.safetensors")):
                    logger.debug(
                        "Skipping non-faster-whisper (safetensors) model directory: "
                        "%s. If this is an MLX/Voxtral build it is handled by its "
                        "own engine; the faster-whisper scanner ignores it.",
                        entry.absolute(),
                    )
                else:
                    logger.warning(
                        "Model directory has no `model.bin` (incomplete download?): "
                        "%s. Ignoring.",
                        entry.absolute(),
                    )
                continue

            self.models[entry.name] = WhisperModel(
                name=entry.name, path=entry.absolute()
            )
