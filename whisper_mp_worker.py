import gc
import os
import traceback
from dataclasses import asdict, is_dataclass
from i18n import t


def whisper_proc_entrypoint(args: dict, q):
    """
    Runs in a child process. Streams progress/logs to parent via `q`.
    Messages put on `q` are dicts with one of the following shapes:
      {"type": "log", "level": "info"|"warn"|"error"|"debug", "msg": "..."}
      {"type": "progress", "pct": float, "detail": "..."}   # optional
      {"type": "result", "ok": True, "segments": [...], "info": {...}}
      {"type": "result", "ok": False, "error": str, "trace": str}
    """
    try:
        # Import heavy libs only in the child
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        import torch
        import yaml
        import i18n

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        # Initialize i18n in child process (PyInstaller uses spawn; no globals shared)
        try:
            app_dir = os.path.abspath(os.path.dirname(__file__))
            i18n.set('filename_format', '{locale}.{format}')
            # Ensure translations directory is available to python-i18n
            i18n.load_path.append(os.path.join(app_dir, 'trans'))
            i18n.set('fallback', 'en')
            # Use locale passed by parent when available
            child_locale = args.get('locale') or 'en'
            i18n.set('locale', child_locale)
        except Exception:
            # Safe fallback: leave i18n defaults; keys may pass through
            pass

        # plog("debug", "Subprocess started. Initializing Whisper model...")

        # Build model in child using provided options
        model = WhisperModel(
            args["model_name_or_path"],
            device=args.get("device", "cpu"),
            compute_type=args.get("compute_type", "float16"),
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )
        # plog("debug", "Model loaded in subprocess")

        # Define callbacks that forward to parent via queue (not used by faster-whisper directly, but kept for parity)
        def log_cb(level, msg):
            plog(level, msg)

        # Prepare audio and VAD
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

        sampling_rate = model.feature_extractor.sampling_rate
        audio = decode_audio(audio_path, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate
        log_cb("info", t('vad'))

        # VAD options
        vad_threshold = float(args.get("vad_threshold", 0.5))
        try:
            vad_parameters = VadOptions(min_silence_duration_ms=500, threshold=vad_threshold, speech_pad_ms=50)
        except TypeError:
            vad_parameters = VadOptions(min_silence_duration_ms=500, onset=vad_threshold, speech_pad_ms=50)

        # Language handling
        language_name = args.get("language_name")
        language_code = args.get("language_code")
        multilingual = False
        whisper_lang = None
        if language_name == "Multilingual":
            multilingual = True
            whisper_lang = None
        elif language_name == "Auto":
            whisper_lang = None
        else:
            whisper_lang = language_code

        # Detect language if requested (Auto)
        if language_name == "Auto":
            language, language_probability, _ = model.detect_language(
                audio, vad_filter=True, vad_parameters=vad_parameters
            )
            log_cb("info", t('language_detect', lang=language, prob=f'{language_probability:.2f}'))
            whisper_lang = language

        # Build prompt/hotwords if disfluencies suppression is requested
        prompt = ""
        if args.get("disfluencies", False):
            try:
                with open(os.path.join(app_dir, 'prompt.yml'), 'r', encoding='utf-8') as f:
                    prompts = yaml.safe_load(f) or {}
                prompt = prompts.get(whisper_lang, '')
            except Exception:
                prompt = ""

        # Perform transcription (streaming)
        segments, info = model.transcribe(
            audio_path,
            language=whisper_lang,
            multilingual=multilingual,
            beam_size=args.get("beam_size", 5),
            # temperature=args.get("temperature"),
            word_timestamps=args.get("word_timestamps", True),
            # initial_prompt=args.get("initial_prompt"),
            hotwords=prompt,
            vad_filter=args.get("vad_filter", True),
            vad_parameters=vad_parameters,
        )
        
        log_cb('info', t('start_transcription') + '\n')
        
        # Stream segments to parent as they arrive
        for s in segments:
            try:
                seg_d = {
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "text": getattr(s, "text", None),
                }
                words = getattr(s, "words", None)
                if words:
                    seg_d["words"] = [
                        {
                            "word": getattr(w, "word", None),
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None),
                            "prob": getattr(w, "probability", None),
                        }
                        for w in words
                    ]
                q.put({"type": "segment", "segment": seg_d})
            except Exception:
                # Best-effort; continue on serialization issues
                pass

        # info into dict
        if is_dataclass(info):
            info_dict = asdict(info)
        else:
            info_dict = {}
            for k in ("language", "language_probability", "duration", "sample_rate"):
                if hasattr(info, k):
                    info_dict[k] = getattr(info, k)
        # Ensure duration is available
        info_dict.setdefault("duration", duration)

        try:
            q.put({"type": "result", "ok": True, "info": info_dict})
        except Exception:
            pass

        # Cleanup VRAM (harmless on CPU)
        try:
            del model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        plog("debug", "Subprocess finished cleanly.")

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
