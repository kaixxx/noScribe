import dataclasses
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
    """

    name: str
    path: Path


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

            # Check here whether a `model.bin` file is present in
            # the directory. This is necessary for a whisper model.
            if not (entry / "model.bin").exists():
                logger.warning(
                    "Missing `model.bin` in model dir: %s. Ignoring.",
                    entry.absolute(),
                )
                continue

            self.models[entry.name] = WhisperModel(
                name=entry.name, path=entry.absolute()
            )
