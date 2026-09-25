import gc
import logging
import os
import platform
import sys
import traceback
from dataclasses import asdict, is_dataclass

if sys.version_info >= (3, 12):
    import importlib.resources as impres
else:
    import importlib_resources as impres

logger = logging.getLogger(__name__)


def whisper_proc_entrypoint(args: dict, q):
    """
    Runs in a child process. Streams progress/logs to parent via `q`.
    Messages put on `q` are dicts with one of the following shapes:
      {"type": "log", "level": "info"|"warn"|"error"|"debug", "msg": "..."}
      {"type": "status", "level": str, "message_id": str, "params": {...}}
      {"type": "progress", "pct": float, "detail": "..."}   # optional
      {"type": "result", "ok": True, "segments": [...], "info": {...}}
      {"type": "result", "ok": False, "error": str, "trace": str}
    """
    try:
        # Import heavy libs only in the child
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions
        import torch
        import yaml
        def status(message_id, params=None, level="info"):
            try:
                q.put({
                    "type": "status",
                    "level": level,
                    "message_id": message_id,
                    "params": params or {},
                })
            except Exception:
                pass
        
        # determine device
        device = args.get("device", "")
        if device != 'cpu':
            if platform.system() == "Darwin":  # MAC
                device = 'auto'
            elif platform.system() in ('Windows', 'Linux'):
                try:
                    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
                except:
                    device = 'cpu'
            else:
                raise Exception('Platform not supported yet.')
            
        # Build model in child using provided options
        model = WhisperModel(
            str(args["model_path"]),
            device=device,
            compute_type=args.get("compute_type", "float16"),
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )

        # Prepare audio and VAD
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

        status("vad")

        # VAD options
        vad_threshold = float(args.get("vad_threshold", 0.5))
        try:
            vad_parameters = VadOptions(min_silence_duration_ms=500, threshold=vad_threshold, speech_pad_ms=50)
        except TypeError:
            vad_parameters = VadOptions(min_silence_duration_ms=500, onset=vad_threshold, speech_pad_ms=50)

        # Language handling
        whisper_lang = args.get("language")
        multilingual = bool(args.get("multilingual", False))
        
        if not model.model.is_multilingual and whisper_lang != 'en':
            whisper_lang = 'en'
            multilingual = False
            status("language_en_only")
        
        if multilingual:
            whisper_lang = None

        # Detect language if requested (Auto)
        if whisper_lang is None and not multilingual:
            audio = decode_audio(
                audio_path, sampling_rate=model.feature_extractor.sampling_rate
            )
            try:
                whisper_lang, language_probability, _ = model.detect_language(
                    audio, vad_filter=True, vad_parameters=vad_parameters
                )
            finally:
                del audio
                gc.collect()
            status("language_detect", {
                "lang": whisper_lang,
                "prob": f'{language_probability:.2f}',
            })

        # Build prompt/hotwords if disfluencies suppression is requested
        prompt = ""
        if args.get("include_disfluencies", False):
            prompt_file = impres.files("prompts") / "prompt.yml"
        else:
            prompt_file = impres.files("prompts") / "prompt_nd.yml"
        try:
            with prompt_file.open("r", encoding="utf-8") as f:
                prompt = yaml.safe_load(f).get(whisper_lang, "")
        except Exception as e:
            logger.exception(e)
            status("err_loading_prompt", level="error")

        # Pass the path so faster-whisper can release its original waveform
        # after VAD, before allocating the full spectrogram (see 28650f2e).
        segments, info = model.transcribe(
            audio_path,
            language=whisper_lang,
            multilingual=multilingual,
            beam_size=args.get("beam_size", 5),
            # temperature=args.get("temperature"),
            word_timestamps=args.get("word_timestamps", True),
            # initial_prompt=prompt,
            hotwords=prompt,
            vad_filter=args.get("vad_filter", True),
            vad_parameters=vad_parameters,
        )
        
        status("start_transcription")
        
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
