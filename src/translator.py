"""Simple translation utilities for noScribe PyQt6 prototype."""
from typing import Dict

# Supported languages and their translation mappings
_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "Audio File": {"de": "Audiodatei"},
    "Transcript File": {"de": "Transkriptdatei"},
    "Start": {"de": "Start"},
    "Stop": {"de": "Stopp"},
    "Select Audio File": {"de": "Audiodatei auswählen"},
    "Select Transcript File": {"de": "Transkriptdatei auswählen"},
    "Selected audio file: {name}": {"de": "Ausgewählte Audiodatei: {name}"},
    "No audio file selected": {"de": "Keine Audiodatei ausgewählt"},
    "Selected transcript file: {name}": {
        "de": "Ausgewählte Transkriptdatei: {name}"
    },
    "No transcript file selected": {
        "de": "Keine Transkriptdatei ausgewählt"
    },
    "Transcription started": {"de": "Transkription gestartet"},
    "Transcription stopped": {"de": "Transkription gestoppt"},
    "Transcription finished": {"de": "Transkription abgeschlossen"},
    "Please select audio and transcript file": {
        "de": "Bitte Audio- und Transkriptdatei wählen"
    },
    "Language": {"de": "Sprache"},
    "Diarization": {"de": "Diarisierung"},
    "English": {"de": "Englisch"},
    "German": {"de": "Deutsch"},
    "Start Time": {"de": "Startzeit"},
    "Stop Time": {"de": "Stoppzeit"},
    "Invalid time format: {time}": {"de": "Ungültiges Zeitformat: {time}"},
}

_language: str = "en"


def set_language(lang: str) -> None:
    """Set current language if supported."""
    global _language
    if lang in ("en", "de"):
        _language = lang


def t(key: str, **kwargs) -> str:
    """Translate a string key with optional format kwargs."""
    template = _TRANSLATIONS.get(key, {}).get(_language, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
