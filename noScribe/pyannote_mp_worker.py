import contextlib
import os
import platform
import sys
import traceback

if sys.version_info >= (3, 12):
    import importlib.resources as impres
else:
    import importlib_resources as impres

if platform.system() == "Darwin" and platform.machine() == "x86_64":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # temp workaround for iomp5 dup

def load_waveform(audio_file):
    """Load audio as the in-memory ``(channels, frames)`` float32 tensor +
    sample rate that pyannote's waveform input expects.

    The input is the WAV written by ``noScribe.audio.convert.ToWav``, which
    owns the format, so plain soundfile can read it -- no torchaudio/torchcodec
    decoding backends needed. Passing the waveform in memory also keeps
    pyannote's own decoder out of play.
    """
    # Both imports are deferred so this module stays stdlib-only at import
    # time, as whisper_mp_worker does. For torch that is load-bearing beyond
    # tidiness: the OMP/MKL environment above must be set before torch pulls in
    # OpenMP, and main.py imports this module in the GUI process just to reach
    # the entrypoint.
    import soundfile
    import torch
    try:
        data, sample_rate = soundfile.read(audio_file, dtype="float32", always_2d=True)
    except RuntimeError as e:
        # libsndfile funnels every open failure through one exception type and
        # only the code tells them apart, so a locked or unreadable file must
        # not be reported as a format problem. (RuntimeError rather than
        # soundfile.LibsndfileError: the latter only exists from soundfile
        # 0.11, and it derives from RuntimeError anyway.)
        if getattr(e, "code", None) == 1:      # SF_ERR_UNRECOGNISED_FORMAT
            raise RuntimeError(
                f"Could not decode {audio_file}: not a WAV the diarization "
                f"worker can read. {e}") from e
        raise RuntimeError(f"Could not read {audio_file}: {e}") from e
    if data.shape[0] == 0:
        # A header with no frames: soundfile accepts it and would hand pyannote
        # a (1, 0) tensor, which its own validator rejects with a message about
        # tensor layout that names nothing the user can act on.
        raise RuntimeError(
            f"{audio_file} contains no audio. Check the start and stop times.")
    # .contiguous() is a no-op for mono (the (frames, 1) transpose is already
    # contiguous); it only copies in the hypothetical multichannel case.
    return torch.from_numpy(data.T).contiguous(), sample_rate  # (ch, frames)


@contextlib.contextmanager
def hide_speechbrain():
    """Keep SpeechBrain out of reach while the diarization pipeline is loaded.

    SpeechBrain is an optional pyannote embedding backend that noScribe's
    bundled pipeline does not use. Older source installations may still have an
    incompatible SpeechBrain version installed, and pyannote's optional import
    would then fail before our pipeline is loaded. Hiding it turns any failure
    into the ImportError pyannote expects, so it records the backend as
    unavailable.

    It has to cover ``Pipeline.from_pretrained``: importing ``Pipeline`` does not
    reach SpeechBrain, because pyannote's package init is lazy and the module
    with the optional import is only loaded when ``from_pretrained`` resolves
    the pipeline class named in config.yaml.
    """
    previous_speechbrain = sys.modules.get("speechbrain")
    sys.modules["speechbrain"] = None
    try:
        yield
    finally:
        if previous_speechbrain is None:
            sys.modules.pop("speechbrain", None)
        else:
            sys.modules["speechbrain"] = previous_speechbrain


# The shortest stretch of audio an embedding is taken from: a shorter span is
# widened evenly to this. 1.0 s reaches into the neighbour: on the tuning pool it
# left 3546 words with the wrong speaker instead of 3505 (faster-whisper).
EMBED_MIN_S = 0.5


# The embed call uses the GPU on Apple hardware only from this much memory on.
# Apple's GPU memory *is* the system memory, so installed RAM is the right thing
# to ask about. CUDA is left alone: its memory is the card's own, free again by
# the time this runs, and its allocator keeps no graph per input length.
EMBED_MPS_MIN_RAM_GB = 16


def _ram_gb():
    """Installed memory in GB, 0 when the platform will not say."""
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
    except (AttributeError, OSError, ValueError):
        return 0


def _centroids(diarization):
    """The pipeline's own speaker centroids as {label: [float]}, or {}.

    They come for free with the diarization (rows follow ``labels()``). A label
    can lack one -- more labels than clusters pads the matrix with zeros -- and a
    recording without speech has none at all; both simply leave voice_check
    with nothing to compare against.
    """
    import numpy as np
    matrix = getattr(diarization, "speaker_embeddings", None)
    if matrix is None:
        return {}
    out = {}
    for label, row in zip(diarization.speaker_diarization.labels(), matrix):
        row = np.asarray(row, dtype="float32")
        if np.all(np.isfinite(row)) and np.any(row):
            out[str(label)] = row.tolist()
    return out


def _embed_spans(pipeline, waveform, sample_rate, spans, q):
    """One embedding per [start_s, end_s] span, None where it cannot be had.

    Raises when not one span could be embedded: that is a check that did not
    run, and returned as a list of None it read like a check that found every
    speaker right ("0 passage(s) reassigned")."""
    import numpy as np
    # pyannote has no public accessor, so the attribute may be gone one day.
    model = getattr(pipeline, "_embedding", None)
    if model is None:
        raise RuntimeError("the diarization pipeline has no embedding model (_embedding)")
    # The crops below bypass the pipeline's own resampling, so they are only
    # valid at the model's rate -- which is what audio.convert.ToWav writes.
    if getattr(model, "sample_rate", sample_rate) != sample_rate:
        raise RuntimeError(f"the embedding model runs at {model.sample_rate} Hz, "
                           f"the audio at {sample_rate} Hz")
    frames = waveform.shape[1]
    shortest = int(EMBED_MIN_S * sample_rate)
    out, last_pct = [], -1
    failed, first_failure = 0, None
    for i, (start_s, end_s) in enumerate(spans):
        a, b = int(start_s * sample_rate), int(end_s * sample_rate)
        missing = shortest - (b - a)
        if missing > 0:
            a -= missing // 2
            b += missing - missing // 2
        a, b = max(0, a), min(frames, b)
        vector = None
        if b - a >= shortest // 2:  # less is left only at the very edge of the file
            try:
                vector = np.asarray(model(waveform[:1, a:b][None]))[0]
                # NaN, or all zeros, is no voice either: a zero vector has no
                # direction, so every cosine with it is undefined and the
                # check would keep each passage as if it had been confirmed
                # (the centroids leave such rows out for the same reason).
                if not (np.all(np.isfinite(vector)) and np.any(vector)):
                    raise ValueError("the model returned no usable vector (NaN or zeros)")
                vector = vector.tolist()
            except Exception as e:
                vector = None  # one span the model rejects must not cost all the others
                failed += 1
                first_failure = first_failure or f"{type(e).__name__}: {e}"
        out.append(vector)
        pct = int((i + 1) / len(spans) * 100)
        if pct != last_pct:
            last_pct = pct
            try:
                q.put({"type": "progress", "step": "voice_check", "pct": pct})
            except Exception:
                pass
    if spans and not any(out):
        raise RuntimeError(f"no span could be embedded"
                           + (f", the first failed with {first_failure}" if first_failure else ""))
    if failed:  # one line, not one per span: the same failure tends to repeat
        try:
            q.put({"type": "log", "level": "warn",
                   "msg": f"Voice check: {failed} of {len(spans)} span(s) could not be embedded, "
                          f"the first with {first_failure}"})
        except Exception:
            pass
    return out


def pyannote_proc_entrypoint(args: dict, q):
    """Runs diarization in a child process and streams progress/logs.
    Messages:
      {"type":"log","level":"info|warn|error|debug","msg":str}
      {"type":"progress","step":str,"pct":int}
      {"type":"result","ok":True,"segments":[{"start":ms,"end":ms,"label":str}],
                                 "centroids":{label:[float]}}
      {"type":"result","ok":False,"error":str,"trace":str}

    With ``args["embed_spans"]`` (a list of ``[start_s, end_s]``) the call does
    not diarize: it returns one speaker embedding per span, from the pipeline's
    own embedding model, for noScribe.voice_check:
      {"type":"result","ok":True,"embeddings":[[float]|None]}
    """
    device = ''
    try:
        import torch
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
           torch.set_num_threads(1)        
        # pyannote's stack (torchmetrics, pyannote's own plotting helpers)
        # imports matplotlib, and matplotlib builds its font cache on first
        # import by asking the OS for every installed font.  noScribe never
        # draws a plot, so the scan is pure cost: 28 s on a cold cache on an
        # M1 against 4 s without it, and on macOS it crashes with
        # KeyError: '_items' when `system_profiler SPFontsDataType` returns
        # an incomplete record (matplotlib#32328, fixed only from matplotlib
        # 3.12).  MPL_IGNORE_SYSTEM_FONTS (honoured from matplotlib 3.11)
        # limits the cache to matplotlib's bundled fonts.  That cache is keyed
        # by font-cache schema version rather than its contents, so it must not
        # land in the user-wide ~/.matplotlib, where another matplotlib process
        # could take the fonts-less list as authoritative; noScribe's own cache
        # directory keeps it private, and an inherited MPLCONFIGDIR is
        # overridden for the same reason.  The font setting uses setdefault so
        # a deliberate override is respected.
        import appdirs
        os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
        os.environ["MPLCONFIGDIR"] = os.path.join(
            appdirs.user_cache_dir("noScribe"), "matplotlib")
        from pyannote.audio import Pipeline

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        class SimpleProgressHook:
            def __init__(self):
                self.step_name = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
                if completed is None:
                    completed = total = 1
                pct = int(completed / total * 100) if total else 100
                if pct > 100:
                    pct = 100
                try:
                    q.put({"type": "progress", "step": str(step_name), "pct": pct})
                except Exception:
                    pass

        audio_file = args.get("audio_path")
        num_speakers = args.get("num_speakers")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)

        plog("debug", "Subprocess (diarize) started. Initializing PyAnnote pipeline...")
        
        # determine xpu
        device = args.get("device", "")
        if device != 'cpu':
            if platform.system() == "Darwin":  # MAC
                device = 'mps' if platform.mac_ver()[0] >= '12.3' and torch.backends.mps.is_available() else 'cpu'
            elif platform.system() in ('Windows', 'Linux'):
                try:
                    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
                except:
                    device = 'cpu'
            else:
                raise Exception('Platform not supported yet.')

        # Every unit is embedded at its own length, and MPS compiles and keeps one
        # graph per distinct input length, so its memory grows with the recording.
        # Measured on 4.7 h of meetings (7044 units, 390 lengths) on an M1 Max: MPS
        # 132 s and 5.1 GB, CPU 225 s and 2.0 GB; on 50 minutes 38 s and 3.3 GB
        # against 45 s and 1.3 GB. About 20 s saved per hour of audio is welcome
        # where 3 GB are to spare and not worth swapping for where they are not.
        # (torch.mps.empty_cache() holds the memory down but makes every call slower
        # than the CPU; the embeddings are the same either way.)
        if args.get("embed_spans") is not None and device == 'mps' and _ram_gb() < EMBED_MPS_MIN_RAM_GB:
            device = 'cpu'

        with impres.as_file(impres.files("pyannote")) as mypath, hide_speechbrain():
            pipeline = Pipeline.from_pretrained(mypath)
        waveform, sample_rate = load_waveform(audio_file)
        pipeline.to(torch.device(device))

        if args.get("embed_spans") is not None:
            embeddings = _embed_spans(pipeline, waveform, sample_rate, args["embed_spans"], q)
            q.put({"type": "result", "ok": True, "embeddings": embeddings})
            return

        seg_list = []
        with SimpleProgressHook() as hook:
            if num_speakers is not None:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook, num_speakers=num_speakers)
            else:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

        for turn, speaker in diarization.speaker_diarization:
            seg_list.append({
                'start': int(turn.start * 1000),
                'end': int(turn.end * 1000),
                'label': speaker,
            })

        try:
            centroids = _centroids(diarization)
        except Exception as e:  # optional: never let them cost the diarization
            plog("warn", f"No speaker centroids: {e}")
            centroids = {}
        try:
            q.put({"type": "result", "ok": True, "segments": seg_list, "centroids": centroids})
        except Exception:
            pass

    except Exception as e:
        try:
            error_str = f"{type(e).__name__}: {e}"
            error_str += f' (device_{device[:3]})' # device_cpu or device_cud or device_mps
            q.put({
                "type": "result",
                "ok": False,
                "error": error_str,
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass

