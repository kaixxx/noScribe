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


def pyannote_proc_entrypoint(args: dict, q):
    """Runs diarization in a child process and streams progress/logs.
    Messages:
      {"type":"log","level":"info|warn|error|debug","msg":str}
      {"type":"progress","step":str,"pct":int}
      {"type":"result","ok":True,"segments":[{"start":ms,"end":ms,"label":str}]}
      {"type":"result","ok":False,"error":str,"trace":str}
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

        with impres.as_file(impres.files("pyannote")) as mypath, hide_speechbrain():
            pipeline = Pipeline.from_pretrained(mypath)
        waveform, sample_rate = load_waveform(audio_file)
        pipeline.to(torch.device(device))

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
            q.put({"type": "result", "ok": True, "segments": seg_list})
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

