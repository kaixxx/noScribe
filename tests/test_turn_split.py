"""Tests for the speaker-turn-aware loop split and the escalation diagnosis.

The cleanest place to cut a looping chunk is where the speaker changes:
context within a turn stays intact, across the change it matters least.
These tests pin the boundary selection (different-speaker only, no overlap
cuts, middle half, nearest-to-middle wins), the window clipping used by the
recursion, and that the ladder actually prefers the turn boundary over the
quietest-frame fallback -- plus the diarization profile line once a loop
resists the gentle repairs.
"""
import numpy as np

from noScribe.voxtral_engine import (
    SAMPLE_RATE,
    _clip_turns,
    _transcribe_guarded,
    _turn_gap_split,
    _turn_profile,
)


def _audio(seconds, silences=()):
    """Noisy audio with 1 s of silence centred on each given second mark."""
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)
    for at in silences:
        a[int((at - 0.5) * SAMPLE_RATE):int((at + 0.5) * SAMPLE_RATE)] = 0.0
    return a


# --------------------------------------------------------------------------- #
# _turn_gap_split
# --------------------------------------------------------------------------- #
def test_cut_lands_on_the_speaker_change():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 99.5, "A"), (100.5, 200.0, "B")]
    cut = _turn_gap_split(audio, turns)
    assert cut is not None
    assert 99.0 <= cut / SAMPLE_RATE <= 101.0

def test_same_speaker_pause_is_not_a_turn_boundary():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 99.5, "A"), (100.5, 200.0, "A")]
    assert _turn_gap_split(audio, turns) is None

def test_overlapping_speech_boundary_is_skipped():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 105.0, "A"), (95.0, 200.0, "B")]  # 10 s overlap
    assert _turn_gap_split(audio, turns) is None

def test_boundary_outside_the_middle_half_is_ignored():
    audio = _audio(200, silences=(20.5,))
    turns = [(0.0, 20.0, "A"), (21.0, 200.0, "B")]  # change at 10% of the window
    assert _turn_gap_split(audio, turns) is None

def test_nearest_boundary_to_the_middle_wins():
    audio = _audio(200, silences=(60.5, 100.0))
    turns = [(0.0, 60.0, "A"), (61.0, 99.5, "B"), (100.5, 200.0, "C")]
    cut = _turn_gap_split(audio, turns)
    assert 99.0 <= cut / SAMPLE_RATE <= 101.0

def test_no_turns_means_no_turn_split():
    assert _turn_gap_split(_audio(10), None) is None
    assert _turn_gap_split(_audio(10), []) is None


# --------------------------------------------------------------------------- #
# _clip_turns / _turn_profile
# --------------------------------------------------------------------------- #
def test_clip_turns_shifts_to_window_relative_times():
    turns = [(10.0, 30.0, "A"), (40.0, 80.0, "B")]
    assert _clip_turns(turns, 20.0, 60.0) == [(0.0, 10.0, "A"), (20.0, 40.0, "B")]
    assert _clip_turns(turns, 90.0, 120.0) is None
    assert _clip_turns(None, 0.0, 10.0) is None

def test_turn_profile_counts_speakers_changes_and_coverage():
    turns = [(0.0, 30.0, "A"), (30.0, 60.0, "B")]
    p = _turn_profile(turns, 60.0)
    assert "2 speaker(s)" in p and "1 turn change(s)" in p and "100% speech" in p


# --------------------------------------------------------------------------- #
# Ladder integration
# --------------------------------------------------------------------------- #
VARIED = ("Der Test zeigt dir schwarz auf weiß, wo du stehst. "
          "Wir haben lange überlegt und uns dann entschieden.")


class _FakeVox:
    """Loops on windows > 150 s (any temperature); short windows come out fine."""

    def __init__(self):
        self.calls = []

    def transcribe_array(self, audio, language, max_new_tokens=0,
                         repetition_penalty=1.0, token_cb=None,
                         temperature=0.0, seed=None, info=None):
        dur = len(audio) / SAMPLE_RATE
        self.calls.append(round(dur))
        if repetition_penalty == 1.0 and dur > 150:
            return "Und dann. " + "Jetzt. " * 300
        return VARIED


def test_ladder_prefers_the_turn_boundary_over_the_middle():
    # Turn change at 130 s, well away from the middle. The quietest-frame
    # fallback only searches +/-5 s around 100 s, so 130/70 halves prove the
    # turn boundary was used.
    audio = _audio(200, silences=(100.0, 130.0))
    turns = [(0.0, 129.5, "A"), (130.5, 200.0, "B")]
    vox = _FakeVox()
    out = _transcribe_guarded(vox, audio, "de", None, "t", turns=turns)
    assert VARIED in out
    assert vox.calls == [200, 200, 130, 70]  # greedy, T=0.2, then the two halves

def test_diagnosis_line_appears_when_gentle_repairs_fail():
    audio = _audio(60)  # too short to split
    turns = [(0.0, 30.0, "A"), (30.0, 60.0, "B")]

    class _AlwaysLoops(_FakeVox):
        def transcribe_array(self, *a, **kw):
            super().transcribe_array(*a, **kw)
            return "Und dann. " + "Jetzt. " * 300

    logs = []
    _transcribe_guarded(_AlwaysLoops(), audio, "de",
                        lambda lvl, msg: logs.append(msg), "t", turns=turns)
    assert any("diarization profile" in m and "2 speaker(s)" in m for m in logs)


def test_alignment_pool_serves_the_loop_ladder():
    """Keeping the prefix needs a time mapping. It hung as a closure on a
    variable in transcribe() and silently became a no-op when the alignment
    architecture was moved onto the pool -- the cherry-pick went through
    without conflicts, the tests stayed green, the feature was dead. Now it
    belongs to the pool, and that is pinned here."""
    from noScribe.voxtral_engine import _AlignerPool

    pool = _AlignerPool("de", None)
    seen = {}

    class _StubAligner:
        def align_prefix(self, words, audio, t_offset=0.0):
            seen["words"], seen["audio"] = words, audio
            seen["how"] = "align_prefix"
            return [{"word": w, "start": i, "end": i + 1, "prob": -0.1}
                    for i, w in enumerate(words)]

        def align_words(self, words, audio, t_offset=0.0, words_span_audio=True):
            seen["how"] = "align_words"
            return []

    def _pick(text, remember=True):
        seen["remember"] = remember
        return _StubAligner()

    pool.aligner_for = _pick
    out = pool.align_for_salvage(["Guten", "Morgen."], "AUDIO")
    assert seen["words"] == ["Guten", "Morgen."] and seen["audio"] == "AUDIO"
    assert out[-1]["end"] == 2
    # The prefix only covers the start of the window. `align_words` cannot do
    # that: it has no way of stopping early and drags the last words across
    # the remaining audio (measured: an 85 s prefix ended at 299.98 s of a
    # 300 s window, with immaculate scores).
    assert seen["how"] == "align_prefix"
    # ...and an internal alignment of a fragment that may well be discarded
    # must not set the model choice for the next chunk.
    assert seen["remember"] is False


def test_salvage_alignment_does_not_move_the_pool_state(monkeypatch):
    """Regression: the salvage rung went through aligner_for and thereby
    overwrote _last_model -- the next chunk without a dominant language of its
    own silently inherited the language of a fragment that often does not
    even end up in the transcript. The same applied to the one-off language
    warning."""
    from noScribe.voxtral_engine import _AlignerPool, ALIGN_MODELS

    loaded = []

    class _Stub:
        def align_prefix(self, words, audio, t_offset=0.0):
            return [{"word": w, "start": 0.0, "end": 1.0, "prob": -0.1}
                    for w in words]

    def _fake_load(self, model, remember=True):
        loaded.append(model)
        if remember:
            self._last_model = model
        return _Stub()

    monkeypatch.setattr(_AlignerPool, "_load", _fake_load)

    # _detect_language needs >= 40 tokens, otherwise the test would run empty.
    german = ("der die das und ich nicht ist wir ein eine mit auf für aber "
              "auch dann wenn noch dass sind habe schon mal jetzt ") * 2
    english = (("the and you that this not with for have are was but they "
                "what just like know then would been your about ") * 2).split()
    from noScribe.voxtral_engine import _detect_language
    assert _detect_language(german)[0] == "de"
    assert _detect_language(" ".join(english))[0] == "en"

    # Auto: chunk 1 is German and sets the choice.
    pool = _AlignerPool(None, None)
    pool.aligner_for(german)
    assert pool._last_model == ALIGN_MODELS["de"]
    # An English salvage passage must not change that.
    pool.align_for_salvage(english, "AUDIO")
    assert pool._last_model == ALIGN_MODELS["de"], \
        "the fragment overwrote the model choice of the next chunk"

    # Explicit language: the one-off warning belongs to the transcript.
    warned = []
    pool2 = _AlignerPool("de", lambda lvl, msg: warned.append((lvl, msg)))
    pool2.align_for_salvage(english, "AUDIO")
    assert not warned, "the warning was spent on a fragment"
    assert pool2._warned is False


def test_transcribe_hands_the_ladder_a_real_callback():
    """Counter-check on the wiring: transcribe() has to pass the pool method
    on, not None and not a stale local variable."""
    import inspect
    from noScribe import voxtral_engine as v

    src = inspect.getsource(v.transcribe)
    call = src[src.index("_transcribe_guarded("):]
    assert "align_cb=(aligner_pool.align_for_salvage" in call, \
        "the ladder no longer gets a time mapping"


def test_backchannel_inside_a_turn_does_not_move_the_cut():
    """A short turn nested inside a longer one must not become a boundary.

    pyannote's standard pattern for an "mhm" during someone's sentence is a
    0.3 s turn whose start and end both fall inside a much longer one. Sorted
    by start time, the turn that follows then looks like it succeeds the
    backchannel, and comparing against *that* turn's end put the cut in the
    middle of the enclosing speaker's utterance -- here at ~22.7 s, in the
    middle of A's 10-30 s turn, instead of at the real change at 30 s.
    """
    audio = _audio(60, silences=(30,))
    turns = [(0.0, 30.0, "A"), (15.0, 15.4, "B"), (30.0, 60.0, "A")]
    # A resumes after B's backchannel -- there is no speaker change at all.
    assert _turn_gap_split(audio, turns) is None


def test_change_after_a_backchannel_cuts_at_the_enclosing_turns_end():
    """With a real change after the nested turn, the cut belongs at its end."""
    audio = _audio(60, silences=(30,))
    turns = [(0.0, 30.0, "A"), (15.0, 15.4, "B"), (30.0, 60.0, "C")]
    cut = _turn_gap_split(audio, turns)
    assert cut is not None
    assert 29.0 <= cut / SAMPLE_RATE <= 31.0
