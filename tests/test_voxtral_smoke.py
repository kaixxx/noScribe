"""Integration smoke test for the Voxtral stack.

Guards the whole pipeline -- mlx-voxtral model load, VoxtralProcessor, the Whisper
encoder, _merge_input_embeddings, and our generate_step decode path -- against a
silent break when the (pinned) mlx / mlx-lm / mlx-voxtral versions or the model
format shift under an environment change. It does NOT assert specific German text
(no bundled/private audio); it asserts the invariant that actually matters:

    the fast greedy path is byte-identical to the library generate() path,

on this machine's real model and library versions, plus "runs without error and
returns a str". Deterministic synthetic audio -- no files, no private data.
(The two paths intentionally diverge only on a single-token repetition loop --
see _consume_tokens -- which this benign tone signal does not trigger.)

Skipped unless mlx is importable AND the local mini-8bit build is present, so it
runs only where someone deliberately has the model (e.g. the dev machine) and is
silently skipped everywhere else.
"""
import os
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_voxtral")

_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "models", "voxtral-mini-8bit")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_MODEL),
    reason="local models/voxtral-mini-8bit not present (smoke test is opt-in)",
)


def _synthetic_audio(seconds=3.0):
    """A short, quiet, deterministic 16 kHz mono signal -- enough to exercise the
    encoder + decoder without needing an audio file."""
    import numpy as np
    from noScribe.voxtral_engine import SAMPLE_RATE
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    # a couple of soft tones; content is irrelevant, determinism is not
    return (0.05 * np.sin(2 * np.pi * 220 * t)
            + 0.03 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)


@pytest.fixture(scope="module")
def voxtral():
    """One loaded model for the whole file. Loading per test would hold two
    6 GB builds at once on a machine the suite is expected to run on."""
    from noScribe.voxtral_engine import _Voxtral
    return _Voxtral(_MODEL)


def test_voxtral_stack_runs_and_fast_matches_library(voxtral):
    from noScribe.voxtral_engine import SAMPLE_RATE

    v = voxtral
    audio = _synthetic_audio()

    # fast path (generate_step) via the public method
    fast = v.transcribe_array(audio, "de")
    assert isinstance(fast, str)

    # library reference path (same greedy settings)
    inp = v.proc.apply_transcrition_request(audio=audio, language="de",
                                            sampling_rate=SAMPLE_RATE)
    mi = {"input_ids": inp.input_ids, "input_features": inp.input_features}
    out = v.model.generate(**mi, max_new_tokens=4096, temperature=0.0,
                           repetition_penalty=1.0)
    ref = v.proc.decode(out[0, inp.input_ids.shape[1]:],
                        skip_special_tokens=True).strip()

    assert fast == ref, (
        "fast generate_step path diverged from library generate() -- a pinned "
        "mlx/mlx-lm/mlx-voxtral upgrade may have changed decode semantics.\n"
        f"fast: {fast[:200]!r}\nref : {ref[:200]!r}"
    )


def test_the_penalty_rung_still_reaches_the_library(voxtral):
    """The retry ladder's repetition-penalty rung is the one production path
    that calls model.generate(), and the only caller passing `stop_tokens=`.
    Nothing else here exercises it, so an upstream rename of that keyword would
    surface at runtime -- on a retry, i.e. on a pass that was already going
    wrong. Content is irrelevant; that the call survives the pin is not."""
    out = voxtral.transcribe_array(_synthetic_audio(), "de",
                                   repetition_penalty=1.2)
    assert isinstance(out, str)
