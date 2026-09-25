"""Command-line entry point for the standalone noScribe server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_server_config
from .processor import RegistryWorkflowProcessor, create_server_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("noscribe-server.yml"),
        help="Path to the server YAML configuration.",
    )
    args = parser.parse_args()
    config = load_server_config(args.config)
    os.umask(0o077)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass
    registry = create_server_registry(config)
    processor = RegistryWorkflowProcessor(
        registry, max_audio_seconds=config.max_audio_hours * 3600
    )
    app = create_app(config, processor)
    try:
        uvicorn.run(app, host=config.host, port=config.port, access_log=False)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
