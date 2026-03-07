import os
import platform
import traceback
from dataclasses import asdict, is_dataclass
import os
import torchaudio
from pathlib import Path

if platform.system() == "Darwin" and platform.machine() == "x86_64":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # temp workaround for iomp5 dup

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
        import yaml
        import torch
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
           torch.set_num_threads(1)        
        from pyannote.audio import Pipeline
        from tempfile import TemporaryDirectory

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
        app_dir = os.path.abspath(os.path.dirname(__file__))
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

        pipeline = Pipeline.from_pretrained(Path(os.path.join(app_dir, 'pyannote')))
        waveform, sample_rate = torchaudio.load(audio_file)        
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

        # ------------------------------------------------------------------
        # Extract per-speaker voice embeddings so noScribe can match them
        # against stored signatures for automatic speaker identification.
        # Wrapped in try/except so a failure here never blocks transcription.
        # ------------------------------------------------------------------
        speaker_embeddings = {}
        try:
            import numpy as np

            # Try to get the embedding model from pipeline internals first
            # (avoids loading a second copy of the model into memory).
            embedding_inference = None
            for _attr in ('_embedding', 'embedding_', '_embedding_model'):
                embedding_inference = getattr(pipeline, _attr, None)
                if embedding_inference is not None:
                    plog("debug", f"Using pipeline.{_attr} for speaker embeddings")
                    break

            # Fallback: load the embedding model directly from disk
            if embedding_inference is None:
                plog("debug", "pipeline._embedding not found – loading embedding model from disk")
                try:
                    from pyannote.audio import Inference, Model
                    emb_model_path = Path(os.path.join(app_dir, 'pyannote', 'embedding', 'pytorch_model.bin'))
                    if emb_model_path.exists():
                        emb_model = Model.from_pretrained(emb_model_path)
                        emb_model.to(torch.device(device))
                        embedding_inference = Inference(emb_model, window="whole")
                        plog("debug", "Loaded embedding model from disk")
                    else:
                        plog("debug", f"Embedding model not found at {emb_model_path}")
                except Exception as load_err:
                    plog("debug", f"Could not load embedding model from disk: {load_err}")

            if embedding_inference is not None:
                # Collect all segments per speaker label
                speaker_windows = {}
                for turn, _, label in diarization.itertracks(yield_label=True):
                    speaker_windows.setdefault(label, []).append(turn)

                plog("debug", f"Extracting embeddings for {len(speaker_windows)} speaker(s)")

                for label, windows in speaker_windows.items():
                    # Use up to 5 longest segments that are at least 1.5 s
                    best = sorted(windows, key=lambda w: w.duration, reverse=True)
                    embs = []
                    for window in best[:5]:
                        if window.duration < 1.5:
                            continue
                        try:
                            start_s = int(window.start * sample_rate)
                            end_s   = int(window.end   * sample_rate)
                            seg_wav = waveform[:, start_s:end_s]
                            raw = embedding_inference(
                                {"waveform": seg_wav, "sample_rate": sample_rate}
                            )
                            # raw may be a SlidingWindowFeature or ndarray
                            if hasattr(raw, 'data'):
                                arr = np.mean(raw.data, axis=0)
                            else:
                                arr = np.array(raw, dtype=np.float32).flatten()
                            if arr.ndim > 1:
                                arr = arr.mean(axis=0)
                            arr = arr.astype(np.float32)
                            if arr.size > 0 and not np.any(np.isnan(arr)):
                                embs.append(arr)
                        except Exception as seg_err:
                            plog("debug", f"Embedding segment error ({label}): {seg_err}")

                    if embs:
                        avg = np.mean(np.stack(embs), axis=0)
                        n = float(np.linalg.norm(avg))
                        if n > 1e-6:
                            avg = avg / n
                        speaker_embeddings[label] = avg.tolist()
                        plog("debug", f"Embedding extracted for {label} ({len(embs)} segments)")
                    else:
                        plog("debug", f"No valid embedding segments for {label}")
            else:
                plog("debug", "No embedding model available – skipping embedding extraction")

        except Exception as emb_err:
            plog("debug", f"Speaker embedding extraction failed: {emb_err}")

        try:
            q.put({"type": "result", "ok": True, "segments": seg_list,
                   "embeddings": speaker_embeddings})
        except Exception:
            pass

    except Exception as e:
        try:
            error_str = f"{type(e).__name__}: {e}"
            error_str += f' (device_{device[:3]})' # device_cpu or device_cud or device_mps
            import traceback as tb
            q.put({
                "type": "result",
                "ok": False,
                "error": error_str,
                "trace": tb.format_exc(),
            })
        except Exception:
            pass

