"""Prepare an opt-in noScribe VOICE edition release build.

Run this script from the repository root AFTER installing runtime dependencies
and huggingface_hub. It downloads multilingual Faster-Whisper Small and creates
an OS-specific release spec based on noScribe's maintained PyInstaller specs.

This does not alter upstream .spec files or existing model directories.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

MODEL_REPO = "Systran/faster-whisper-small"
ROOT = Path(__file__).resolve().parents[2]


def replace_once(source: str, before: str, after: str, label: str) -> str:
    n = source.count(before)
    if n != 1:
        raise RuntimeError(
            f"Cannot safely patch {label}: expected one source block, got {n}. "
            "The upstream spec has changed; review it before continuing."
        )
    return source.replace(before, after, 1)


def create_spec(platform: str) -> Path:
    spec_dir = ROOT / "pyinstaller"
    if platform == "windows":
        template = spec_dir / "noScribe_win.spec"
        output = spec_dir / "noScribe_voice_win.spec"
    elif platform == "macos":
        template = spec_dir / "noScribe_macOS.spec"
        output = spec_dir / "noScribe_voice_macOS.spec"
    else:
        template = spec_dir / "noScribe_linux.spec"
        output = spec_dir / "noScribe_voice_linux.spec"

    if not template.exists():
        raise FileNotFoundError(f"Official noScribe spec not found: {template}")
    text = template.read_text(encoding="utf-8")

    if platform == "windows":
        text = replace_once(
            text,
            "('../models/precise/', './models/precise/'),\n"
            "('../models/fast/', './models/fast/'),\n"
            "('../noScribeEdit/', './noScribeEdit/'),",
            "('../models/small/', './models/small/'),",
            "Windows bundled models/editor",
        )
    elif platform == "macos":
        text = replace_once(
            text,
            "('../models/precise', 'models/precise/'), "
            "('../models/fast', 'models/fast/'),",
            "('../models/small', 'models/small/'),",
            "macOS bundled models",
        )
    else:
        start_marker = "# Add model files to `datas` if available."
        end_marker = "\na = Analysis("
        start = text.find(start_marker)
        end = text.find(end_marker, start)
        if start < 0 or end < 0 or text.count(start_marker) != 1:
            raise RuntimeError("Linux model bundle stanza changed upstream; not editing")
        replacement = (
            "# VOICE edition: bundle the multilingual Faster-Whisper Small model.\n"
            "if not (Path('models') / 'small' / 'model.bin').is_file():\n"
            "    raise FileNotFoundError('models/small/model.bin not present')\n"
            "datas.append(('../models/small', 'models/small/'))\n"
        )
        text = text[:start] + replacement + text[end:]

    output.write_text(text, encoding="utf-8")
    print(f"Generated {output.relative_to(ROOT)}")
    return output


def prepare_model() -> None:
    from huggingface_hub import snapshot_download

    target = ROOT / "models" / "small"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading multilingual {MODEL_REPO} to {target}...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(target),
        allow_patterns=["*.bin", "*.json", "*.txt", "README.md", "LICENSE*"],
    )
    model_bin = target / "model.bin"
    if not model_bin.is_file() or model_bin.stat().st_size < 100_000_000:
        raise RuntimeError("Whisper Small model.bin missing or incomplete")
    if not (target / "config.json").is_file():
        raise RuntimeError("Whisper Small config.json missing")
    # Hugging Face's download-cache metadata does not belong in release assets.
    shutil.rmtree(target / ".cache", ignore_errors=True)
    print(f"Verified model.bin ({model_bin.stat().st_size / (1024**2):.0f} MiB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=("windows", "macos", "linux"), required=True)
    args = p.parse_args()
    if not (ROOT / "noScribe" / "__main__.py").is_file():
        raise SystemExit(f"Run from a noScribe repository containing source: {ROOT}")
    prepare_model()
    create_spec(args.platform)


if __name__ == "__main__":
    main()
