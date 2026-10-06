import contextlib
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

from i18n import t

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
        import i18n

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        # Initialize python-i18n in child process (PyInstaller uses spawn; no
        # globals shared).
        #
        # TODO: python-i18n is unmaintained for more than five years.
        #
        # See the main file for more information on python-i18n and the
        # approach. Here nothing should actually fail as any possible
        # exceptions were already handled/checked in the main app.
        i18n.set("filename_format", "{locale}.{format}")
        i18n.set("enable_memoization", True)
        i18n.set("fallback", "en")
        i18n.set("locale", args.get("locale", "en"))

        with impres.as_file(impres.files("trans")) as mypath:
            i18n.load_path.append(mypath)

            # Using `t` once here to load the localization files into memory.
            # As there is no `print`, nothing happens really.
            t("app_header")
        
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
            str(args["whisper_model"].path),
            device=device,
            compute_type=args.get("compute_type", "float16"),
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )

        # Define callbacks that forward to parent via queue (not used by faster-whisper directly, but kept for parity)
        def log_cb(level, msg):
            plog(level, msg)

        # Prepare audio and VAD
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

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
        
        if not model.model.is_multilingual and language_code != 'en':
            language_name = 'English'
            language_code = 'en'
            log_cb("info", t('language_en_only'))
        
        if language_name == "Multilingual":
            multilingual = True
            whisper_lang = None
        elif language_name == "Auto":
            whisper_lang = None
        else:
            whisper_lang = language_code

        speech_turns = args.get("speech_turns")

        # Detect language if requested (Auto)
        if language_name == "Auto":
            audio = decode_audio(
                audio_path, sampling_rate=model.feature_extractor.sampling_rate
            )
            try:
                with _speech_map_from(speech_turns):
                    whisper_lang, language_probability, _ = model.detect_language(
                        audio, vad_filter=True, vad_parameters=vad_parameters
                    )
            finally:
                del audio
                gc.collect()
            log_cb("info", t('language_detect', lang=whisper_lang, prob=f'{language_probability:.2f}'))

        # Build prompt/hotwords if disfluencies suppression is requested
        prompt = ""
        if args.get("disfluencies", False):
            prompt_file = impres.files("prompts") / "prompt.yml"
        else:
            prompt_file = impres.files("prompts") / "prompt_nd.yml"
        try:
            with prompt_file.open("r", encoding="utf-8") as f:
                prompt = yaml.safe_load(f).get(whisper_lang, "")
        except Exception as e:
            logger.exception(e)
            log_cb('error', t('err_loading_prompt') + '\n')

        # Pass the path so faster-whisper can release its original waveform
        # after VAD, before allocating the full spectrogram (see 28650f2e).
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
            log_cb("debug", "speech map from the diarization unavailable; Silero decides where speech is")
        
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
