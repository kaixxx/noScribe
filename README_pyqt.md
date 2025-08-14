# PyQt6 Prototype

This repository contains an experimental PyQt6 interface for **noScribe**.
The GUI layout is defined in `ui/noScribe.ui` and loaded at runtime by
`src/main.py`. Styles are provided via `resources/style.qss`.

Run the prototype from the repository root with:

```bash
python src/main.py
```

The interface currently implements basic file selection and logging and is
intended as a starting point for further porting from the previous
customtkinter implementation.

## Localization

A simple translation helper (`src/translator.py`) demonstrates how UI strings
can be localized. By default the prototype uses English, but the language can
be switched by calling `set_language("de")` before creating the main window.
