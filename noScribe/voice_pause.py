"""Optional VOICE-style pause markers.

Based on the user's noScribe 0.7 pause formatting. Measures silent gaps
between consecutive ASR segments after existing VAD boundary adjustment.
"""

def format_voice_pause(gap_seconds: float) -> str | None:
    if 0.2 <= gap_seconds <= 0.6:
        return " (.)"
    if gap_seconds > 0.6:
        return f" ({max(1, round(gap_seconds))})"
    return None
