#!/bin/sh
set -e

# `gui` as first argument starts the Tk interface instead; it needs an X
# server passed into the container (see README).
if [ "${1:-}" = "gui" ]; then
  shift
  exec python -m noScribe "$@"
fi

# Default: run headless. Without --no-gui noScribe would try to open a Tk
# window and fail for lack of a display. Appending the flag is harmless for
# --help and --help-models.
exec python -m noScribe "$@" --no-gui
