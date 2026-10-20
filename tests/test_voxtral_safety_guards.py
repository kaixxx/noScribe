"""The guard rails before model loading -- and those of the salvage cut.

What they have in common is that a mistake here does not crash: it sends the
machine into swap, never lets the job finish, or removes speech from the
transcript, without the log looking any different from a good run.
"""
import numpy as np
import pytest

from noScribe import voxtral_engine as v
from noScribe.voxtral_engine import (
    HARD_MIN_CHUNK_SEC,
    SAMPLE_RATE,
    _model_kind,
    _salvage_prefix,
    _transcribe_guarded,
)


class _StopRun(Exception):
    """The stubbed model constructor raises this: proves that the run got
    exactly as far as model loading and not one step further."""


def _tiny_wav(tmp_path):
    import soundfile as sf
    p = tmp_path / "t.wav"
    sf.write(p, np.zeros(2 * 16000, dtype="float32"), 16000)
    return str(p)


def _sized(monkeypatch, tmp_path, ram=32.0, **kw):
    """Drive transcribe() up to model loading and collect the log lines."""
    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: ram)
    logs = []
    with pytest.raises(_StopRun):
        v.transcribe(_tiny_wav(tmp_path), need_timestamps=False,
                     log_cb=lambda lvl, msg: logs.append((lvl, msg)), **kw)
    return logs


# --------------------------------------------------------------------------- #
# Memory profile of a build
# --------------------------------------------------------------------------- #
def test_a_24b_build_without_a_bit_width_is_metered_conservatively():
    """`small` is the 4-bit profile and therefore the cheapest of the three 24B
    entries. An unquantised or 6-bit folder without a bit width in its name
    landed on it and got passes several hundred seconds long for weights that
    need 25-48 GB -- the run then swaps endlessly instead of being refused."""
    assert _model_kind("models/Voxtral-Small-24B-2507") == "small8"
    assert _model_kind("my-voxtral-24b") == "small8"
    # With an explicit bit width everything stays as it was.
    assert _model_kind("voxtral-small-4bit") == "small"
    assert _model_kind("voxtral-small-6bit") == "small6"
    assert _model_kind("voxtral-small-8bit") == "small8"
    # The mini side was never affected: without a bit width it hits the more
    # expensive bf16 profile, i.e. the safe direction.
    assert _model_kind("models/Voxtral-Mini-3B-2507") == "mini"
    assert _model_kind("voxtral-mini-8bit") == "mini8"


def test_an_unquantised_source_release_is_refused_by_path_too(monkeypatch, tmp_path):
    """The block only compared the full Hub repo ID. A local copy under
    models/ is a file path, though -- so exactly the build the block is meant
    to stop got past it."""
    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 64.0)
    wav = _tiny_wav(tmp_path)
    for repo in ("mistralai/Voxtral-Small-24B-2507",
                 "models/Voxtral-Small-24B-2507",
                 "/srv/builds/Voxtral-Mini-3B-2507"):
        with pytest.raises(ValueError, match="unquantised source release"):
            v.transcribe(wav, voxtral_repo=repo, need_timestamps=False)


# --------------------------------------------------------------------------- #
# Pinned chunk_sec
# --------------------------------------------------------------------------- #
def test_a_sub_second_pinned_chunk_sec_cannot_collapse_the_passes(monkeypatch, tmp_path):
    """int() turns anything below 1 s into 0, and max(1, 0 * SAMPLE_RATE)
    turns that into a pass of ONE sample: a 2-minute file fell apart into 2401
    Voxtral decodes of 50 ms each, and the job never finished. Until now the
    floor applied only on the automatic path."""
    logs = _sized(monkeypatch, tmp_path, chunk_sec=0.5,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert any(lvl == "warn" and "below the" in msg and "minimum" in msg
               for lvl, msg in logs), logs
    # ...and the old, misleading justification is gone.
    assert not any("more context than the model has" in msg for _, msg in logs), logs


def test_a_pinned_chunk_sec_below_the_floor_is_lifted(monkeypatch, tmp_path):
    logs = _sized(monkeypatch, tmp_path, chunk_sec=30,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert any(str(HARD_MIN_CHUNK_SEC) in msg for _, msg in logs), logs


def test_a_sane_pinned_chunk_sec_is_left_alone(monkeypatch, tmp_path):
    """Counter-check: a usable value must not trigger a warning."""
    logs = _sized(monkeypatch, tmp_path, chunk_sec=600,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert not any(lvl == "warn" and "voxtral_chunk_sec" in msg
                   for lvl, msg in logs), logs


# --------------------------------------------------------------------------- #
# What the salvage cut may rely on
# --------------------------------------------------------------------------- #
LOOP_TEXT = ("Wir haben gestern lange über die eigentlichen Ziele gesprochen. "
             "Danach kam ziemlich unvermittelt die Frage nach den Werten auf. "
             "Ich fand diese Diskussion ausgesprochen aufschlussreich. "
             + "dass das, " * 40).strip()


def _stamps(words, real_tail, step=4.0):
    """One word every `step` seconds -- far enough apart that the prefix lies
    above SALVAGE_MIN_PREFIX_SEC and the shortness bound does not fire first.
    `real_tail` decides whether the LAST word was really aligned or is only
    an estimate."""
    out = [{"word": w, "start": i * step, "end": (i + 1) * step, "prob": -0.1}
           for i, w in enumerate(words)]
    if not real_tail:
        out[-1] = dict(out[-1], prob=0.0)
    return out


def test_salvage_refuses_a_cut_taken_from_an_invented_timestamp():
    """The check used any() over the whole prefix -- a single genuinely
    aligned word was enough. The cut, however, is taken exclusively on the
    LAST timestamp. When that time was guessed (evenly spread or interpolated
    words at the end), the resume point landed a measured 27 s too late, and
    the speech in between fell out of the transcript."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)

    good, why = _salvage_prefix(
        LOOP_TEXT, audio, lambda w, a: _stamps(w, real_tail=True))
    assert good is not None and why is None

    bad, why = _salvage_prefix(
        LOOP_TEXT, audio, lambda w, a: _stamps(w, real_tail=False))
    assert bad is None
    assert "not really aligned" in why, why


def test_salvage_still_refuses_a_fully_spread_alignment():
    """The older case is kept, together with its justification in the log."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)
    spread = lambda w, a: [{"word": x, "start": float(i), "end": float(i + 1),
                            "prob": 0.0} for i, x in enumerate(w)]
    out, why = _salvage_prefix(LOOP_TEXT, audio, spread)
    # Evenly spread times are pure guesswork; the justification has to name
    # that, otherwise a refused salvage cannot be told in the log from a rung
    # that never ran.
    assert out is None and "even guess" in why


def test_a_degenerate_prefix_is_not_shipped_as_a_success():
    """The `rest_sec < 1.0` exit was the only success exit of the ladder
    without a degeneracy check: a prefix that is still looping itself passed
    as a finished result and never escalated."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)
    # A prefix whose sentence repeats verbatim (the compression arm of the
    # detector), followed by a short cycle loop.
    sentence = "Und dann sagte sie genau dasselbe noch ein weiteres Mal dazu. "
    text = (sentence * 60 + "dass das, " * 40).strip()

    calls = []

    class _Vox:
        def transcribe_array(self, audio, language, max_new_tokens=4096,
                             repetition_penalty=1.0, token_cb=None,
                             temperature=0.0, seed=None, info=None):
            calls.append(round(len(audio) / SAMPLE_RATE))
            return text if len(calls) == 1 else "Ein sauberer zweiter Versuch."

    def _align_to_the_very_end(words, window):
        # Last word really aligned, but practically at the window end ->
        # rest_sec < 1.0, i.e. the exit in question.
        n = len(words)
        step = (len(window) / SAMPLE_RATE) / n
        return [{"word": w, "start": i * step, "end": (i + 1) * step,
                 "prob": -0.1} for i, w in enumerate(words)]

    logs = []
    out = _transcribe_guarded(_Vox(), audio, "de",
                              lambda lvl, msg: logs.append(msg), "Pass 1/1",
                              align_cb=_align_to_the_very_end)
    assert not v._looks_degenerate(out), out[:120]
    assert len(calls) > 1, "the ladder did not carry on after the prefix"
    assert any("still looks degenerate" in m for m in logs), logs


def test_the_salvage_alignment_runs_on_a_cleared_mlx_cache():
    """The prefix rung aligns right after the looping decode -- typically the
    longest of the job -- and the aligner's forward runs on the same GPU. The
    main path frees the dead MLX buffers before its alignment; this one did
    not, which is exactly the co-residency MIN_HEADROOM_GB is meant to rule out."""
    events = []

    class _Mx:
        def clear_cache(self):
            events.append("clear")

    class _Vox:
        _mx = _Mx()

        def transcribe_array(self, audio, language, max_new_tokens=4096,
                             repetition_penalty=1.0, token_cb=None,
                             temperature=0.0, seed=None, info=None):
            events.append("decode")
            return LOOP_TEXT if events.count("decode") == 1 else "Ein sauberer Rest."

    def _align(words, window):
        events.append("align")
        return _stamps(words, real_tail=True)

    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)
    _transcribe_guarded(_Vox(), audio, "de", None, "Pass 1/1", align_cb=_align)
    assert "align" in events, events
    assert events[events.index("align") - 1] == "clear", events
