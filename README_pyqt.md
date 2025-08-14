# PyQt6 Prototype

This repository contains a PyQt6 interface for **noScribe**.
The GUI layout is defined in `ui/noScribe.ui` and loaded at runtime by
`src/main.py`. Styles are provided via `resources/style.qss`.

Run the application from the repository root with:

```bash
python src/main.py
```

The interface features a header with the application logo, a sidebar for
selecting input and output files, language and diarization options,
start/stop time fields and start/stop buttons. Transcription is performed in a
background thread using `faster-whisper` and progress messages are appended to
the log view.

## Localization

A simple translation helper (`src/translator.py`) demonstrates how UI strings
can be localized. The language can be switched at runtime via the combo box in
the sidebar.
