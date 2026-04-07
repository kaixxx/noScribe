import dataclasses
import importlib.resources as impres
import logging
from pathlib import Path

import huggingface_hub

from . import exception

PATH_PACKAGE_MODELS = impres.files("models")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WhisperModel:
    """
    Represents a whisper model or more specifically a model that can be used
    for transcriptions.
    """

    name: str
    path: Path


class WhisperModelCollector:
    """
    Collects available models in either the package directory or in given
    additional paths.
    """

    def __init__(self, additional_paths: list | None = None):
        self.models: dict = {}
        self.paths: Path = {PATH_PACKAGE_MODELS}

        if additional_paths:
            for item in additional_paths:
                self.paths.add(item)

        self.update()

    def update(self):
        for path in self.paths:
            if not path.is_dir():
                logger.warning("Given model path is not a directory: %s.", path)
                continue

            for entry in path.iterdir():
                if not entry.is_dir():
                    continue

                if entry.name in self.models:
                    logger.warning(
                        "Found duplicate model: %s (%s).",
                        entry.name,
                        entry.absolute(),
                    )
                    continue

                # Check here whether a `model.bin` file is present in
                # the directory. This is necessary for a whisper model.
                if not (entry / "model.bin").exists():
                    logger.warning(
                        "Missing `model.bin` in model dir: %s.",
                        entry.absolute(),
                    )
                    continue

                self.models[entry.name] = WhisperModel(
                    entry.name, entry.absolute()
                )

    def get_names(self) -> set[str]:
        return self.models.keys()

    def get_paths(self) -> set[Path]:
        ret = {item.path for item in self.models.values()}
        return ret

    def get_path_for(self, model_name: str) -> Path:
        ret = self.models.get(model_name, None)
        if ret:
            ret = ret.path

        return ret


class WhisperModelDownloader:
    # Define a dict of available models.
    # TODO: store this somewhere else in the future.
    models: dict = {
        "precise": {
            "repository": "dropbox-dash/faster-whisper-large-v3-turbo",
            "subfolder": None,
            "files": [
                "README.md",
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ],
            "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        },
        "fast": {
            "repository": "mukowaty/faster-whisper-int8",
            "subfolder": "faster-whisper-large-v3-turbo-int8",
            "files": [
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ],
            "revision": "db1cdb5990df067cf53670d86241330c7ea454cf",
        },
        "ci": {
            "repository": "mukowaty/faster-whisper-int8",
            "subfolder": "faster-whisper-tiny-int8",
            "files": [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.json",
            ],
            "revision": "db1cdb5990df067cf53670d86241330c7ea454cf",
        },
    }

    def __init__(self, path_target: Path):
        self.path_target = path_target

    def get_avail_models(self) -> set[str]:
        return self.models.keys()

    def download(self, model: str, force: bool = False):
        # Check whether directory is already present. If that's case and
        # `force` is false, fail.
        if (self.path_target / model).exists() and not force:
            raise exception.ModelAlreadyExists(model)

        # Raise error if the desired model does not exist.
        try:
            tmp = self.models[model]
        except KeyError as e:
            raise exception.ModelDoesNotExist(model) from e

        for file in tmp["files"]:
            local_file: str = huggingface_hub.hf_hub_download(
                repo_id=tmp["repository"],
                filename=file,
                subfolder=tmp["subfolder"],
                local_dir=self.path_target / model,
            )

            # Move files one level up if `subfolder` is given.
            if tmp["subfolder"]:
                local_file = Path(local_file)
                local_file.rename(self.path_target / model / local_file.name)


# Helper functions
# TODO: remove these after refactoring the main code.
def initialize_user_whisper_model_dir(path_user_models: Path):
    path_user_models.mkdir(exist_ok=True)
    path_readme = path_user_models / "README.txt"

    if not path_readme.exists():
        with open(path_readme, "w") as fd:
            fd.write(
                "You can download custom Whisper-models for the transcription\n"
                "into this folder. See here for more information:\n"
                "https://github.com/kaixxx/noScribe/wiki/Add-custom-Whisper-models-for-transcription"
            )
