import contextlib
import gc
import logging
import os
import platform
import sys
from dataclasses import asdict, is_dataclass

if sys.version_info >= (3, 12):
    import importlib.resources as impres
else:
    import importlib_resources as impres

from .worker_errors import WorkerState

logger = logging.getLogger(__name__)

# Gaps closed and edges padded by the same amounts as the Silero options in
# whisper_proc_entrypoint, and kept equal on purpose -- but measured for this map on
# its own: closing gaps up to 2 s instead let one phone call become a single 31-s
# chunk that Whisper decoded into an invented sentence.
SPEECH_MAP_MIN_GAP_S = 0.5
SPEECH_MAP_PAD_S = 0.05


def speech_map(turns, n_samples, sampling_rate=16000):
    """faster-whisper's speech chunks ({'start', 'end'} in samples) from
    [start_s, end_s] turns: padded by SPEECH_MAP_PAD_S, merged across gaps under
    SPEECH_MAP_MIN_GAP_S, clipped to the audio."""
    spans = sorted((max(0, int((s - SPEECH_MAP_PAD_S) * sampling_rate)),
                    min(n_samples, int((e + SPEECH_MAP_PAD_S) * sampling_rate)))
                   for s, e in turns if e > s)  # an empty turn is no speech, padded or not
    out = []
    for s, e in spans:
        if s >= e:  # a turn that lies past the end of the audio
            continue
        if out and s - out[-1]['end'] < SPEECH_MAP_MIN_GAP_S * sampling_rate:
            out[-1]['end'] = max(out[-1]['end'], e)
        else:
            out.append({'start': s, 'end': e})
    return out


def speech_turns_from(diarization):
    """[start_s, end_s] turns for the speech map from the diarization the job
    hands the worker (DiarizationResult.to_mapping(): milliseconds), or None.

    pyannote's turns only, recognised by the evidence its plugin adds: as the
    speech map, Nemotron's many short turns cost words on the German hand
    references (9.28 % -> 9.78 % WER against Silero; pyannote's 8.75 %)."""
    if not diarization or "local-pyannote" not in (diarization.get("extensions") or {}):
        return None
    return [[seg["start"] / 1000.0, seg["end"] / 1000.0]
            for seg in diarization.get("segments") or []] or None


# With speaker detection, the diarization's turns tell Whisper where speech is,
# in place of Silero's voice activity. Silero drops quiet speech before Whisper
# ever hears it: of a voice 15-20 dB below the one beside it, it kept 27-41 % of
# the time, and Whisper wrote 2-3 % of the words in what it dropped. Lowering its
# threshold does not help (0.35 is within noise on AMI, 0.15 lets room tone in).
# The diarization's turns kept 67 % of that voice, and with them as the speech
# map Whisper wrote 50-65 % of its words instead of 27-46 %, with none in pure
# noise. On all 23 AMI meetings, against their manual words, it made 1.39 points
# fewer errors per reference word (95 % [+0.93, +1.92]), with more words found
# in every meeting; on four hand-checked German passages 9.28 % -> 8.75 % WER,
# all four better. Single files vary between runs either way -- faster-whisper
# re-decodes an unsure window by sampling, unseeded, and quiet speech is where it
# is unsure (one recording gave 219 to 292 words) -- so only the totals say anything.
@contextlib.contextmanager
def _speech_map_from(turns):
    """While active, faster-whisper takes its speech chunks from `turns` instead
    of Silero. It asks for them through the one name it imports into its
    transcribe module, and nothing else there changes: the chunks are collected
    and the timestamps mapped back exactly as with Silero's. Yields whether the
    swap is in place -- not without turns, nor if a faster-whisper release no
    longer has that name (tests/test_whisper_speech_map.py pins it)."""
    import faster_whisper.transcribe as ft
    silero = getattr(ft, "get_speech_timestamps", None)
    if not turns or silero is None:
        yield False
        return
    ft.get_speech_timestamps = lambda audio, *a, **k: speech_map(turns, len(audio))
    try:
        yield True
    finally:
        ft.get_speech_timestamps = silero


def whisper_proc_entrypoint(args: dict, q):
    """
    Runs in a child process. Streams progress/logs to parent via `q`.
    Messages put on `q` are dicts with one of the following shapes:
      {"type": "log", "level": "info"|"warn"|"error"|"debug", "msg": "..."}
      {"type": "status", "level": str, "message_id": str, "params": {...}}
      {"type": "progress", "pct": float, "detail": "..."}   # optional
      {"type": "result", "ok": True, "segments": [...], "info": {...}}
      {"type": "state", "device": str|None, "stage": str}
      Failed results include the structured fields from worker_errors.error_result.
    """
    state = WorkerState(q, args.get('device'))
    try:
        state.update('imports')
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
        
        state.update('device_selection')
        # determine device
        device = args.get("device", "")
        if device in ('', 'auto'):
            if platform.system() == "Darwin":  # MAC
                device = 'auto'
            elif platform.system() in ('Windows', 'Linux'):
                try:
                    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
                except:
                    device = 'cpu'
            else:
                raise Exception('Platform not supported yet.')
            
        state.update('model_loading', device=device)
        # Build model in child using provided options
        model = WhisperModel(
            str(args["model_path"]),
            device=device,
            compute_type=args.get("compute_type", "float16"),
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )
        state.update('audio_loading', device=getattr(model.model, 'device', device))
        q.put({"type": "log", "level": "info", "msg": f"Whisper: using {state.device}"})

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

        speech_turns = speech_turns_from(args.get("diarization"))

        # Detect language if requested (Auto)
        if whisper_lang is None and not multilingual:
            audio = decode_audio(
                audio_path, sampling_rate=model.feature_extractor.sampling_rate
            )
            try:
                state.update('language_detection')
                with _speech_map_from(speech_turns):
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
        state.update('preprocessing')
        # transcribe() takes its speech map before it returns the lazy segments,
        # so the swap need only last for the call.
        with _speech_map_from(speech_turns) as mapped:
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
        if speech_turns and not mapped:
            q.put({"type": "log", "level": "debug",
                   "msg": "speech map from the diarization unavailable; Silero decides where speech is"})
        
        status("start_transcription")
        
        # Stream segments to parent as they arrive
        state.update('transcription')
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
            q.put(state.failure(e))
        except Exception:
            pass
