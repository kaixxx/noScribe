import os
import platform
import traceback
from dataclasses import asdict, is_dataclass
import os

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

        # Expand relative model paths to absolute paths inside app folder
        with open(os.path.join(app_dir, 'pyannote', 'pyannote_config.yaml'), 'r') as yaml_file:
            pyannote_config = yaml.safe_load(yaml_file)
        pyannote_config['pipeline']['params']['embedding'] = os.path.join(
            app_dir, *pyannote_config['pipeline']['params']['embedding'].split("/"))
        pyannote_config['pipeline']['params']['segmentation'] = os.path.join(
            app_dir, *pyannote_config['pipeline']['params']['segmentation'].split("/"))

        tmpdir = TemporaryDirectory('noScribe_diarize')
        tmp_cfg = os.path.join(tmpdir.name, 'pyannote_config_macOS.yaml')
        with open(tmp_cfg, 'w') as yaml_file:
            yaml.safe_dump(pyannote_config, yaml_file)

        pipeline = Pipeline.from_pretrained(tmp_cfg)
        pipeline.to(torch.device(device))

        seg_list = []
        with SimpleProgressHook() as hook:
            if num_speakers is not None:
                diarization = pipeline(audio_file, hook=hook, num_speakers=num_speakers)
            else:
                diarization = pipeline(audio_file, hook=hook)

        for segment, _, label in diarization.itertracks(yield_label=True):
            seg_list.append({
                'start': int(segment.start * 1000),
                'end': int((segment.start + segment.duration) * 1000),
                'label': label,
            })

        try:
            q.put({"type": "result", "ok": True, "segments": seg_list})
        except Exception:
            pass

    except Exception as e:
        try:
            import traceback as tb
            q.put({
                "type": "result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "trace": tb.format_exc(),
            })
        except Exception:
            pass

