"""
Subprocess entry point for the Voxtral transcription backend.

Mirrors the message protocol of `whisper_mp_worker` so the main app can consume
segments the same way regardless of the chosen engine:

    {"type": "log", "level": "info|warn|error|debug", "msg": str}
    {"type": "progress", "pct": int}
    {"type": "segment", "segment": {"start","end","text","words"}}
    {"type": "result", "ok": True, "info": {...}}
    {"type": "result", "ok": False, "error": str, "trace": str}
"""

import traceback


def voxtral_proc_entrypoint(args: dict, q):
    try:
        from noScribe import voxtral_engine

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        def progress(pct):
            try:
                q.put({"type": "progress", "pct": int(pct)})
            except Exception:
                pass

        def send_segment(seg):
            # Deliberately NOT wrapped in try/except: if the queue breaks we
            # must fail the whole job (outer handler -> ok=False) rather than
            # silently truncate the transcript and still report success.
            q.put({"type": "segment", "segment": seg})

        # segment_cb streams each pass's segments as soon as it finishes, so
        # the GUI can display and autosave a partial transcript during long
        # files (and a cancel/crash loses at most the current pass).
        _, info = voxtral_engine.transcribe(
            audio_path=args["audio_path"],
            # None for Auto/Multilingual jobs -> Voxtral auto-detects the
            # language, and the aligner is chosen per chunk from the
            # transcribed text (char-native model when a language dominates,
            # romanised multilingual fallback otherwise).
            language=args.get("language_code"),
            need_timestamps=args.get("need_timestamps", True),
            voxtral_repo=args.get("voxtral_repo"),
            # None/0 -> engine picks the per-pass length from available RAM.
            chunk_sec=args.get("chunk_sec"),
            corrections_path=args.get("corrections_path"),
            # correct spelling of names the user already told us
            speaker_names=args.get("speaker_names"),
            # None -> engine default; lower it on a machine with nothing else running
            ram_reserve_gb=args.get("ram_reserve_gb"),
            # [[start_s, end_s, label], ...] from the diarization, or None.
            # Used to cut looping chunks at speaker-turn boundaries and to log
            # a diarization profile when a loop resists repair.
            speaker_turns=args.get("speaker_turns"),
            log_cb=plog,
            progress_cb=progress,
            segment_cb=send_segment,
        )

        q.put({"type": "result", "ok": True, "info": info})

    except Exception as e:
        try:
            q.put({
                "type": "result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass
