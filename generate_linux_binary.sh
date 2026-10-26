#!/bin/bash
set -euo pipefail

# Builds the Linux binary with PyInstaller inside the build environment from the
# root Dockerfile. The image gets its own tag so it does not replace the
# noScribe image from docker/ (tagged noscribe).
IMAGE=noscribe-pyinstaller

docker buildx build --platform linux/amd64 -t "$IMAGE" --load .
docker run --rm -it --mount type=bind,src=.,dst=/usr/src/app "$IMAGE" pyinstaller pyinstaller/noScribe_linux.spec
