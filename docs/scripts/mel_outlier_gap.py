"""How far above the rest does the loudest Mel cell lie? On real material.

The clamp floor is `log_max - 8`, and `log_max` is **one** cell out of
millions. Whether that is fragile depends solely on how far this one cell lies
above the remaining material: if it coincides with a high percentile, both
floors are identical and the whole question is moot. If it lies far above, the
floor is set by an outlier.

What is measured is therefore the **distance in dB between the maximum and
several percentiles** of the same log-Mel, per file and additionally over
sliding windows of pass length -- because what counts is the outlier *within*
what gets normalised together in one pass.

No model, only signal. Result in dB: a 10 dB gap means the floor sits 10 dB
higher than a robust measure would set it.

    python docs/scripts/mel_outlier_gap.py <audio> [audio ...]
"""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mlx_voxtral.audio_processing import log_mel_spectrogram   # noqa: E402

SR = 16000
PCTS = (99.99, 99.9, 99.0)
WINDOW_SEC = 600.0          # a typical pass length of the chunker


def load16k(path):
    """Any format -> 16 kHz mono float32, via ffmpeg like the pipeline."""
    if str(path).lower().endswith(".wav"):
        a, sr = sf.read(str(path), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr == SR:
            return np.ascontiguousarray(a)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp = fh.name
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", str(path),
                    "-ar", str(SR), "-ac", "1", tmp], check=True)
    a, _ = sf.read(tmp, dtype="float32")
    pathlib.Path(tmp).unlink(missing_ok=True)
    return np.ascontiguousarray(a)


def raw_log_mel(a):
    """Unclamped log-Mel: the `global_max` lever, affinity inverted."""
    cs = 30 * SR
    n = int(np.ceil(len(a) / cs))
    a = np.pad(a, (0, n * cs - len(a)))
    return np.array(log_mel_spectrogram(mx.array(a), global_max=-1e6)) * 4.0 - 4.0


def gaps(mel):
    """Distance maximum - percentile, in dB (one log10 unit = 10 dB)."""
    mx_ = float(mel.max())
    return {p: (mx_ - float(np.percentile(mel, p))) * 10.0 for p in PCTS}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("    python")[1].strip())
    hdr = "  ".join(f"max-p{p:g}" for p in PCTS)
    print(f"{'file':34s} {'duration':>7s}  {hdr}   worst {WINDOW_SEC:.0f}s window")
    for path in sys.argv[1:]:
        a = load16k(path)
        mel = raw_log_mel(a)
        g = gaps(mel)
        # sliding windows: the outlier only counts within one pass
        step = int(WINDOW_SEC * 100)          # 100 Mel frames per second
        worst = 0.0
        for s in range(0, max(1, mel.shape[1] - step + 1), step):
            w = mel[:, s:s + step]
            if w.shape[1] < step // 2:
                continue
            worst = max(worst, gaps(w)[99.9])
        name = pathlib.Path(path).name[:33]
        cells = "  ".join(f"{g[p]:7.1f}" for p in PCTS)
        print(f"{name:34s} {len(a)/SR:6.0f}s  {cells}   {worst:20.1f} dB", flush=True)


if __name__ == "__main__":
    main()
