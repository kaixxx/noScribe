"""Prepare a CPU-only noScribe VOICE macOS Intel (x86_64) build.

Run on a disposable GitHub Actions checkout, not your normal working tree.
This intentionally edits sources in that checkout to disable unavailable local
speaker diarization and generates a dedicated PyInstaller spec without PyAnnote.
The Windows, Linux and Apple Silicon specs remain unchanged.
"""
from __future__ import annotations

import platform
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "Systran/faster-whisper-small"


def replace_exact(text: str, before: str, after: str, label: str) -> str:
    n = text.count(before)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, got {n}; no file written")
    return text.replace(before, after, 1)


def prepare_model() -> None:
    from huggingface_hub import snapshot_download

    target = ROOT / "models" / "small"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(target),
        allow_patterns=["*.bin", "*.json", "*.txt", "README.md", "LICENSE*"],
    )
    model = target / "model.bin"
    if not model.is_file() or model.stat().st_size < 100_000_000:
        raise RuntimeError("Whisper Small model.bin missing or incomplete")
    if not (target / "config.json").is_file():
        raise RuntimeError("Whisper Small config.json missing")
    shutil.rmtree(target / ".cache", ignore_errors=True)
    print(f"Bundling multilingual Whisper Small: {model.stat().st_size // (1024*1024)} MiB")


def edit_sources() -> None:
    """Edit only the CI checkout, leaving the user's repo/source branch unchanged."""
    main_path = ROOT / "noScribe" / "main.py"
    factory_path = ROOT / "noScribe" / "plugins" / "factory.py"

    main = main_path.read_text(encoding="utf-8")
    main = replace_exact(
        main,
        "    speaker_setting = speaker_detection if speaker_detection is not None else 'auto'\n",
        "    speaker_setting = speaker_detection if speaker_detection is not None else 'auto'\n"
        "    # Intel-only bundled edition has no compatible local PyAnnote installation.\n"
        "    if platform.system() == 'Darwin' and platform.machine() == 'x86_64':\n"
        "        speaker_setting = 'none'\n",
        "Disable diarization in Intel transcription jobs",
    )
    main = replace_exact(
        main,
        "        self.option_menu_speaker.set(get_config('last_speaker', 'auto'))\n",
        "        self.option_menu_speaker.set(get_config('last_speaker', 'auto'))\n"
        "        if platform.system() == 'Darwin' and platform.machine() == 'x86_64':\n"
        "            self.option_menu_speaker.configure(values=['none'], state='disabled')\n"
        "            self.option_menu_speaker.set('none')\n",
        "Disable Intel speaker selector in GUI",
    )

    factory = factory_path.read_text(encoding="utf-8")
    factory = replace_exact(
        factory,
        "from pathlib import Path\n",
        "from pathlib import Path\nimport platform\n",
        "Add platform import to backend factory",
    )
    factory = replace_exact(
        factory,
        "    return BackendRegistry([\n"
        "        LocalWhisperPlugin(host, whisper_models, settings_store=store),\n"
        "        LocalPyannotePlugin(host, settings_store=store),\n"
        "    ])\n",
        "    plugins = [LocalWhisperPlugin(host, whisper_models, settings_store=store)]\n"
        "    if not (platform.system() == 'Darwin' and platform.machine() == 'x86_64'):\n"
        "        plugins.append(LocalPyannotePlugin(host, settings_store=store))\n"
        "    return BackendRegistry(plugins)\n",
        "Omit unavailable Intel PyAnnote backend",
    )
    # Only write after every target/source has passed strict validation.
    main_path.write_text(main, encoding="utf-8")
    factory_path.write_text(factory, encoding="utf-8")
    print("Intel edition: removed local PyAnnote availability and disabled speaker selection")


def make_spec() -> None:
    spec_dir = ROOT / "pyinstaller"
    template = spec_dir / "noScribe_macOS.spec"
    output = spec_dir / "noScribe_voice_intel.spec"
    spec = template.read_text(encoding="utf-8")
    spec = replace_exact(
        spec,
        "('../models/precise', 'models/precise/'), ('../models/fast', 'models/fast/'),",
        "('../models/small', 'models/small/'),",
        "Whisper Small data folder in macOS spec",
    )
    spec = replace_exact(
        spec,
        "('../pyannote', 'pyannote/'), ",
        "",
        "Remove PyAnnote model folder",
    )
    spec = replace_exact(
        spec,
        "datas += collect_data_files('lightning_fabric')\n",
        "",
        "Omit Lightning data collection",
    )
    spec = replace_exact(
        spec,
        "tmp_ret = collect_all('pyannote')\n"
        "datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]\n",
        "",
        "Omit PyAnnote binary collection",
    )
    spec = replace_exact(
        spec,
        "excludes=['speechbrain', 'noScribe.server', 'uvicorn'],",
        "excludes=['speechbrain', 'pyannote', 'lightning_fabric', 'torch', "
        "'torchaudio', 'torchcodec', 'noScribe.server', 'uvicorn'],",
        "Exclude unsupported diarization stack",
    )
    spec = replace_exact(
        spec,
        "name='noScribe.app',",
        "name='noScribe-VOICE.app',",
        "Distinct Intel app bundle name",
    )
    output.write_text(spec, encoding="utf-8")
    print(f"Generated {output.relative_to(ROOT)}")


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        raise SystemExit("This release preparation requires native macOS x86_64")
    prepare_model()
    # Validate/construct the spec before editing checked-out application sources.
    make_spec()
    edit_sources()


if __name__ == "__main__":
    main()
