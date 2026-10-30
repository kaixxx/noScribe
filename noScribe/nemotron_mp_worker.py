"""Speaker diarization with NVIDIA's Nemotron 3 Diarization, in a child process.

An alternative to pyannote_mp_worker, run by the local-nemotron plugin
(noScribe/plugins/local_nemotron); main.select_diarization_model picks one.
It answers with the same result message -- turns as
{"start": ms, "end": ms, "label": "SPEAKER_nn"} -- and instead of centroids it
returns the model's own speaker probabilities (one row per FRAME_S, one column
per speaker, float16), which is what the voice check compares against
(noScribe.voice_check.Probabilities). They travel as data, base64 of the
float16 rows, so a server can hand them to the desktop: about 7.7 MB per hour.

Measured against pyannote (issue #360): on 15 of the 16 AMI test meetings (not
in Nemotron's training data; VoxConverse and AMI train/dev are) 11.6 % of words
end up under the wrong speaker against 15.7 %, at about a seventh of the time;
on 40 German CallHome calls 6.8 % of speech time against 11.8 % (CallHome Part 1
is in the training data, whether these calls are is not known); a 4.8 h video
call runs through its speaker cache in 50 s.

What it cannot do: more than eight speakers (it folds the others into them;
main.py says so when all eight are in use), and a fixed number of speakers. It needs transformers with
`nemotron3_diarization`, 5.18 or newer, and librosa, with which transformers
builds the processor's mel filters.

Messages:
  {"type":"log","level":"info|warn|error|debug","msg":str}
  {"type":"progress","step":"diarization","pct":int}
  {"type":"state","device":str|None,"stage":str}
  {"type":"result","ok":True,"segments":[{"start":ms,"end":ms,"label":str}],
   "extensions":{"local-nemotron":{"probabilities":
       {"float16_base64":str,"shape":[rows,columns],"columns":[label],"frame_s":float}}}}
  Failed results include the structured fields from worker_errors.error_result.

Stdlib-only at import time, like the other workers: the heavy imports happen in
the entrypoint.
"""
import os
import sys
import threading

from .worker_errors import WorkerState

if sys.version_info >= (3, 12):
    import importlib.resources as impres
else:
    import importlib_resources as impres

# The Hugging Face repository, used when no copy ships in models/.
MODEL_REPO = "nvidia/Nemotron-3-Diarization"
MODEL_DIR = "nemotron-diarization"  # under models/

# The model's output frame: transformers upsamples its 80 ms encoder frames to
# the 10 ms of the spectrogram.
FRAME_S = 0.01

# The key this worker's own data travels under in DiarizationResult extensions:
# the plugin's backend ID.
EXTENSION_KEY = "local-nemotron"

# A worker without steps of its own reports its whole run under this name.
PROGRESS_STEP = "diarization"

# The model takes the whole recording in one call that reports nothing, 90 s
# for 36 minutes of audio on a CPU and longer for longer recordings. A server
# streams the worker's events to its client, and a reverse proxy cuts a stream
# that stays silent past its idle timeout (nginx: 60 s), so the last progress
# is sent again at this interval while the worker runs.
KEEPALIVE_S = 15
# The share of the progress bar the features take; the model has the rest.
FEATURES_PCT = 20


class Progress:
    """Progress messages under PROGRESS_STEP: sent when they change and, as a
    keep-alive, repeated every KEEPALIVE_S while the block runs."""

    def __init__(self, q):
        self.q, self.pct, self.stop = q, 0, threading.Event()
        self.thread = threading.Thread(target=self._repeat, daemon=True)

    def __call__(self, pct):
        pct = int(pct)
        if pct != self.pct:
            self.pct = pct
            self._send()

    def _send(self):
        try:
            self.q.put({"type": "progress", "step": PROGRESS_STEP, "pct": self.pct})
        except Exception:
            pass

    def _repeat(self):
        while not self.stop.wait(KEEPALIVE_S):
            self._send()

    def __enter__(self):
        self._send()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        return False

# A speaker counts as talking where the model's probability exceeds this: the
# model card's and transformers' default, and what the measurement used.
THRESHOLD = 0.5


def model_source():
    """The shipped copy under models/ if there is one, else the Hub repository."""
    try:
        path = impres.files("models") / MODEL_DIR
        if (path / "config.json").is_file():
            return str(path)
    except Exception:
        pass
    return MODEL_REPO


def weights_on_disk():
    """Whether the model is shipped under models/ or already whole in the Hugging Face
    cache: the worker then loads it without asking the Hub."""
    if model_source() != MODEL_REPO:
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(MODEL_REPO, name), str)
                   for name in ("config.json", "processor_config.json", "model.safetensors"))
    except Exception:
        return False


def installed():
    """Whether the installed transformers knows the model and librosa is there for
    its processor (transformers builds the mel filters with it). Found without
    importing transformers: the GUI process asks this and never loads it."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("transformers")
        if spec is None or not spec.origin:
            return False
        if not os.path.isdir(os.path.join(os.path.dirname(spec.origin), "models", "nemotron3_diarization")):
            return False
        return importlib.util.find_spec("librosa") is not None
    except Exception:
        return False


def available():
    """installed(), and the weights are on disk, so running it starts no download."""
    return installed() and weights_on_disk()


def encode_probabilities(probabilities, columns):
    """The probability rows as JSON-compatible data: base64 of the float16 bytes,
    exactly what the worker computed (decode_probabilities restores them)."""
    import base64
    import numpy as np
    rows = np.ascontiguousarray(probabilities, dtype='<f2')
    return {"float16_base64": base64.b64encode(rows.tobytes()).decode("ascii"),
            "shape": list(rows.shape), "columns": list(columns), "frame_s": FRAME_S}


def decode_probabilities(data):
    """The numpy rows from encode_probabilities' data."""
    import base64
    import numpy as np
    rows = np.frombuffer(base64.b64decode(data["float16_base64"]), dtype='<f2')
    return rows.reshape(data["shape"])


def label(column):
    """The diarization label of the model's speaker column `column`."""
    return f"SPEAKER_{int(column):02d}"


def features(processor, audio, sample_rate, chunk_frames=60000, device="cpu", on_progress=None):
    """The model's input for a whole recording, as processor(audio) gives it, but
    computed ten minutes at a time: in one pass transformers' spectrogram took
    12.3 GB for a 4.8 h recording whose features are 0.9 GB. transformers keeps the
    one pass by design (huggingface/transformers#49090) and points to the
    processor's streaming calls, whose chunks reproduce it frame for frame; only
    the model's forward stays offline, since its streaming mode caches speakers
    differently. What is left after the last whole chunk comes from an offline call
    on the recording's end. The streaming call for a last chunk keeps the last
    valid frame since transformers 5.18 (huggingface/transformers#49167), but in
    this mode without look-ahead it still refuses an end under 416 samples and
    rounds a lone frame differently (up to 9.5e-7): on 5.18.0, 19 of 4,526 lengths
    raised and 179 differed, with chunks of 8 to 8,000 frames. Bit-identical to
    the one pass."""
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    hop = processor.feature_extractor.hop_length
    valid = len(audio) // hop  # frames one pass counts as valid; it adds one more, masked
    step = chunk_frames // processor.subsampling_factor
    processor.streaming_modes = dict(processor.streaming_modes, noscribe_offline=(step, 0))
    processor.set_streaming_mode("noscribe_offline")
    if len(audio) <= processor.num_samples_first_audio_chunk:
        return processor(audio, sampling_rate=sample_rate).to(device)

    feats, done = None, 0
    start, end, first = 0, processor.num_samples_first_audio_chunk, True
    while end <= len(audio):
        part = processor(audio[start:end], sampling_rate=sample_rate, is_streaming=True,
                         is_first_audio_chunk=first).input_features
        if feats is None:
            feats = torch.empty((1, valid + 1, part.shape[2]), dtype=part.dtype, device=device)
        feats[:, done:done + part.shape[1]] = part
        done += part.shape[1]
        if on_progress:
            on_progress(done / valid)
        start = processor.audio_chunk_start(done)
        end, first = start + processor.num_samples_per_audio_chunk, False
    tail = max(0, done - 16) * hop  # on the frame grid, and far enough back for the window and pre-emphasis
    feats[:, done:valid] = processor(audio[tail:], sampling_rate=sample_rate).input_features[:, done - tail // hop:valid - tail // hop]
    feats[:, valid:] = processor.feature_extractor.padding_value
    mask = torch.zeros(feats.shape[:2], dtype=torch.long, device=device)
    mask[:, :valid] = 1
    return BatchFeature({"input_features": feats, "attention_mask": mask})


def turns(speaker_dicts):
    """transformers' [{"Start": s, "End": s, "Speaker": k}] as noScribe's turns,
    sorted by start, which the overlap assignment in main.py relies on."""
    out = [{"start": round(d["Start"] * 1000), "end": round(d["End"] * 1000),
            "label": label(d["Speaker"])} for d in speaker_dicts]
    return sorted((t for t in out if t["end"] > t["start"]), key=lambda t: (t["start"], t["end"]))


def nemotron_proc_entrypoint(args: dict, q):
    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    state = WorkerState(q, args.get('device'))
    try:
        state.update('imports')
        # huggingface_hub sends usage telemetry (library, versions) with its
        # requests unless told not to, and reads this switch once, when it is
        # imported. Keep noScribe processing local, even if the parent
        # environment enables telemetry for other apps.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # First: pyannote_mp_worker sets the OpenMP variables torch has to find at import.
        from .pyannote_mp_worker import select_device
        try:
            from transformers.models import nemotron3_diarization  # noqa: F401
        except ImportError:
            import transformers
            raise RuntimeError(f"Nemotron diarization needs transformers 5.18 or newer; "
                               f"this one is {transformers.__version__}") from None

        audio_file = args["audio_path"]
        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)
        state.update('device_selection')
        device = select_device(args.get("device", ""))

        state.update('model_loading', device=device)
        source = model_source()
        # Weights on disk load without asking the Hub, so a job stays offline
        # and a new upload of the model cannot change results unannounced.
        offline = weights_on_disk()
        plog("debug", f"Loading Nemotron diarization from {source} on {device}")
        with Progress(q) as progress:
            speaker_dicts, probabilities = _diarize(source, offline, audio_file, device, state, progress)
            progress(100)
        q.put({"type": "result", "ok": True, "segments": turns(speaker_dicts),
               "extensions": {EXTENSION_KEY: {"probabilities": encode_probabilities(
                   probabilities, [label(k) for k in range(probabilities.shape[1])])}}})
    except Exception as e:
        try:
            q.put(state.failure(e))
        except Exception:
            pass


def _diarize(source, offline, audio_file, device, state, progress):
    """The turns and the per-frame speaker probabilities of a recording, the
    features reported as the first FEATURES_PCT of the progress."""
    import numpy as np
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    from .pyannote_mp_worker import load_waveform

    processor = AutoProcessor.from_pretrained(source, local_files_only=offline)
    state.update('audio_loading')
    waveform, sample_rate = load_waveform(audio_file)
    audio = waveform[0].numpy()  # ToWav writes mono; a view, not a copy
    del waveform
    if sample_rate != processor.feature_extractor.sampling_rate:
        raise RuntimeError(f"audio at {sample_rate} Hz, the model needs "
                           f"{processor.feature_extractor.sampling_rate} Hz")
    inputs = features(processor, audio, sample_rate, device=device,
                      on_progress=lambda done: progress(done * FEATURES_PCT))
    del audio
    progress(FEATURES_PCT)
    # Loaded after the features, so it is not held through their peak.
    model = AutoModelForAudioFrameClassification.from_pretrained(
        source, local_files_only=offline).to(device).eval()
    inputs = inputs.to(device, dtype=model.dtype)
    # Offline mode: the whole recording in one call, cut into chunks with a
    # speaker cache inside the model, so there is no length limit.
    state.update('diarization')
    with torch.inference_mode():
        logits = model(**inputs).logits
    mask = inputs.attention_mask
    del inputs  # the whole recording's features, on the device
    speaker_dicts = processor.extract_speaker_dict(logits, mask, threshold=THRESHOLD)[0]
    probabilities = torch.sigmoid(logits[0].float()).cpu().numpy().astype(np.float16)
    return speaker_dicts, probabilities
