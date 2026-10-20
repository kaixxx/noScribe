"""Unit tests for the in-place loop breaker and the retry ladder order.

_LoopBreaker is backend-agnostic (tokens need len/indexing, logits item
assignment), so these tests drive it with plain lists and numpy arrays --
no weights, no mlx. The ladder tests script a stub Voxtral to verify the
escalation order: greedy -> gentle sampling -> split -> stronger sampling ->
penalty, with the penalty strictly last.
"""
import numpy as np

from noScribe.voxtral_engine import (
    _LoopBreaker,
    _other_language,
    _transcribe_guarded,
    LOOP_BREAK_MAX_KICKS,
    LOOP_BREAK_MAX_PERIOD,
    LOOP_BREAK_WINDOW,
    RETRY_REPETITION_PENALTIES,
    RETRY_TEMPERATURES,
    SAMPLE_RATE,
)

VOCAB = 100


def _drive(breaker, seq):
    """Feed `seq` step by step the way generate_step calls a logits processor:
    at each step the processor sees the tokens generated so far and the next
    logits row. Returns the set of steps at which a ban was placed."""
    bans = []
    for i in range(len(seq)):
        logits = np.zeros((1, VOCAB), dtype=np.float32)
        out = breaker(seq[:i], logits)
        banned = np.where(np.isinf(out[0]))[0]
        if banned.size:
            bans.append((i, int(banned[0])))
    return bans


# --------------------------------------------------------------------------- #
# _LoopBreaker
# --------------------------------------------------------------------------- #
def test_clean_text_is_never_touched():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, VOCAB, size=2000).tolist()
    br = _LoopBreaker()
    assert _drive(br, seq) == []
    assert br.kicks == 0 and not br.gave_up


def test_short_legitimate_repetition_is_not_flagged():
    # A phrase repeated 5x (35 tokens) then normal text: far below the
    # full-window threshold, must pass untouched ("sehr, sehr" stays safe).
    rng = np.random.default_rng(1)
    seq = [3, 1, 4, 1, 5, 9, 2] * 5 + rng.integers(0, VOCAB, size=1000).tolist()
    br = _LoopBreaker()
    assert _drive(br, seq) == []
    assert br.kicks == 0


def test_loop_gets_banned_at_cycle_continuation():
    cycle = [3, 1, 4, 1, 5, 9, 2]
    seq = cycle * 60  # 420 tokens of pure loop
    br = _LoopBreaker()
    bans = _drive(br, seq)
    assert bans, "loop was never detected"
    first_step, banned_tok = bans[0]
    # Detection needs a full periodic window first.
    assert first_step >= LOOP_BREAK_WINDOW
    # The banned token is exactly the one that would continue the cycle.
    assert banned_tok == seq[first_step - len(cycle)]
    assert br.kicks == 1 and not br.gave_up


def test_repeated_reentry_gives_up():
    # The "model" ignores every kick: after each ban episode one divergent
    # token, then straight back into the loop. After LOOP_BREAK_MAX_KICKS
    # episodes the breaker must give up so the retry ladder takes over.
    cycle = list(range(7))
    seq = []
    for burst in range(LOOP_BREAK_MAX_KICKS + 1):
        seq += cycle * ((LOOP_BREAK_WINDOW // len(cycle)) + 4)
        seq.append(99)  # divergence ends the episode
    br = _LoopBreaker()
    _drive(br, seq)
    assert br.kicks == LOOP_BREAK_MAX_KICKS + 1
    assert br.gave_up


# --------------------------------------------------------------------------- #
# Retry ladder order
# --------------------------------------------------------------------------- #
DEGEN = ("na " * 500).strip()          # word-run 500 -> degenerate
DEGEN_SHORT = ("na " * 40).strip()     # degenerate, but the shortest candidate
CLEAN = "alles gut, kurzer sauberer text"
NO_SPLIT_AUDIO = np.zeros(60 * SAMPLE_RATE, dtype=np.float32)  # < 2*45s


class _ScriptedVox:
    """transcribe_array stub: pops scripted (text, info) results and records
    the decode parameters of every attempt."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def transcribe_array(self, audio, language, max_new_tokens=4096,
                         repetition_penalty=1.0, token_cb=None,
                         temperature=0.0, seed=None, info=None):
        self.calls.append((temperature, repetition_penalty))
        text, extra = self.script.pop(0)
        if info is not None:
            info.update(extra)
        return text


def test_clean_pass_is_single_attempt():
    vox = _ScriptedVox([(CLEAN, {})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out == CLEAN
    assert vox.calls == [(0.0, 1.0)]


def test_inplace_repair_is_accepted_without_retry():
    vox = _ScriptedVox([(CLEAN, {"loop_kicks": 2, "loop_gave_up": False})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out == CLEAN
    assert len(vox.calls) == 1


def test_gave_up_attempt_is_never_trusted():
    # The truncated text looks clean to the text detector (dilution), but the
    # breaker gave up -- the ladder must retry, not accept it.
    vox = _ScriptedVox([
        (CLEAN, {"loop_gave_up": True, "loop_kicks": 4}),
        (CLEAN, {}),
    ])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out == CLEAN
    t0, _ = RETRY_TEMPERATURES[0]
    assert vox.calls == [(0.0, 1.0), (t0, 1.0)]


def test_gentle_sampling_comes_before_any_penalty():
    vox = _ScriptedVox([(DEGEN, {}), (CLEAN, {})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out == CLEAN
    t0, _ = RETRY_TEMPERATURES[0]
    assert vox.calls == [(0.0, 1.0), (t0, 1.0)]


def test_full_ladder_order_and_shortest_fallback():
    # Everything fails; audio too short to split, so the order is:
    # greedy, T=0.2, split, T=0.8, penalty 1.01, penalty 1.1 -- penalty strictly last.
    vox = _ScriptedVox([
        (DEGEN, {}), (DEGEN, {}), (DEGEN, {}), (DEGEN_SHORT, {}), (DEGEN, {}),
    ])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    expected = [(0.0, 1.0)]
    expected += [(t, 1.0) for t, _ in RETRY_TEMPERATURES]
    expected += [(0.0, p) for p in RETRY_REPETITION_PENALTIES]
    assert vox.calls == expected
    assert out == DEGEN_SHORT  # shortest garbage wins when nothing resolves


def test_a_truncated_attempt_never_beats_a_complete_one():
    """When the breaker gives up, _consume_tokens cuts the stream off in the
    middle of the run -- so that attempt is the shortest by construction. As
    long as the ladder simply took the shortest candidate at the end, it won
    every comparison: measured, a 148-word stump went into the transcript
    instead of an almost complete 2733-word transcript, and the rest of the
    chunk was gone."""
    stump = ("Ein sauberer Anfang, der abbricht. " + "na " * 20).strip()
    vox = _ScriptedVox([
        (stump, {"loop_gave_up": True, "loop_kicks": 4}),
        (DEGEN, {}), (DEGEN, {}), (DEGEN, {}), (DEGEN, {}),
    ])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out != stump, "the truncated attempt won again"
    assert out == DEGEN
    assert len(out.split()) > len(stump.split())


def test_a_truncated_attempt_is_used_when_it_is_all_there_is():
    """Counter-check: it must not lead to nothing coming back at all."""
    stump = "nur dieses Bruchstück"
    vox = _ScriptedVox([(stump, {"loop_gave_up": True})] * 5)
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t")
    assert out == stump


def test_incremental_sync_survives_the_tail_bound():
    """The count of tokens seen must not be derived from len(self._tail): the
    buffer stops growing at the bound, so from then on every step looked like
    an offset and took the full resync -- ~220 device scalars per generated
    token, a measured ~10 ms of extra latency per token."""
    class _Tokens(list):
        def __init__(self, items):
            super().__init__(items)
            self.slices = 0

        def __getitem__(self, item):
            if isinstance(item, slice):
                self.slices += 1
            return list.__getitem__(self, item)

    rng = np.random.default_rng(3)
    seq = rng.integers(0, VOCAB, size=3 * LOOP_BREAK_WINDOW).tolist()
    br = _LoopBreaker()
    slices = 0
    for i in range(len(seq)):
        toks = _Tokens(seq[:i])
        br(toks, np.zeros((1, VOCAB), dtype=np.float32))
        slices += toks.slices

    assert slices == 0, f"{slices} full resynchronisations"
    assert br._seen == len(seq) - 1
    # The buffer stays bounded nonetheless.
    assert len(br._tail) <= LOOP_BREAK_WINDOW + LOOP_BREAK_MAX_PERIOD


# --------------------------------------------------------------------------- #
# A window that really is in another language
# --------------------------------------------------------------------------- #
ENGLISH = ("And then you know what they said, that this was not the way we would "
           "have done it, but they just went with it anyway and I think that was "
           "the right call for the team, because you have to trust the people "
           "who are there and not second guess them all the time when they are "
           "the ones who know the work best.")
GERMAN = ("Und dann haben wir das noch mal besprochen, weil ich nicht sicher war, "
          "ob das so passt, aber jetzt ist es klar und wir sind auch schon "
          "weiter, denn wenn die anderen dann noch mal schauen, habe ich das "
          "Gefühl, dass es schon gut wird und wir das jetzt so lassen können, "
          "ohne dass noch mal jemand etwas ändert.")


def test_a_window_really_in_another_language_does_not_climb_the_ladder():
    """A German file with an English passage on Auto (or a wrong pinned
    language): the language check marked every decode bad, no rung could make
    English speech come out German, and the split recursed to its leaves --
    60 decodes of a 600 s window -- before best() shipped the penalty attempt,
    the one rung that deletes real repeated words. Now the gentle retry and one
    split, whose halves stop after their own two decodes, settle it: six
    decodes, four of them half as long. Nor is the prefix rung tried: with no
    loop to cut off it only logged that it could not find one, which read as
    if the pass had looped."""
    audio = np.zeros(600 * SAMPLE_RATE, dtype=np.float32)
    vox = _ScriptedVox([(ENGLISH, {})] * 6)   # a seventh decode would fail
    logged = []
    out = _transcribe_guarded(vox, audio, "de", lambda level, msg: logged.append(msg),
                              "t", want_lang="de", align_cb=lambda words, window: [])
    assert out == f"{ENGLISH} {ENGLISH}"
    t0, _ = RETRY_TEMPERATURES[0]
    assert vox.calls == [(0.0, 1.0), (t0, 1.0)] * 3
    assert not any("clean part" in m for m in logged), logged


def test_the_split_still_repairs_a_translation_that_survives_the_retry():
    """Which language Voxtral writes in hangs on the exact window length, so
    the split -- which changes it -- is the rung most likely to undo a
    translation the gentle retry could not. It still runs once."""
    audio = np.zeros(600 * SAMPLE_RATE, dtype=np.float32)
    vox = _ScriptedVox([(ENGLISH, {}), (ENGLISH, {}), (GERMAN, {}), (GERMAN, {})])
    out = _transcribe_guarded(vox, audio, "de", None, "t", want_lang="de")
    assert out == f"{GERMAN} {GERMAN}"
    assert len(vox.calls) == 4


def test_a_language_the_detector_cannot_name_is_never_called_translated():
    """The script check reads all Cyrillic as Russian, so a correctly pinned
    Ukrainian file came back "translated ('ru' instead of 'uk')" on every
    pass, each one sent up the ladder for nothing. Czech reads as Polish by
    its function words, the same way."""
    ukrainian = ("Ми ще раз усе обговорили, бо я не був певен, чи це підходить, "
                 "але тепер усе зрозуміло, і ми вже рухаємося далі, бо коли інші "
                 "ще раз подивляться, у мене є відчуття, що все буде добре.")
    assert _other_language(ukrainian, "uk") is None
    assert _other_language(ukrainian, "de") == "ru"      # a language it can name
    assert _other_language(ENGLISH, "de") == "en"


def test_english_from_a_language_the_detector_cannot_name_is_still_caught():
    """Silencing every language outside the detector's list also silenced the
    one translation Voxtral makes: a pinned Hindi, Finnish or Korean pass that
    came back in English was neither retried nor reported any more."""
    for want in ("hi", "fi", "ko", "uk", "sv"):
        assert _other_language(ENGLISH, want) == "en", want


def test_a_spurious_translation_is_still_retried():
    """Counter-check: what the language check exists for. A German window that
    came back translated is repaired by the gentle-sampling rung."""
    vox = _ScriptedVox([(ENGLISH, {}), (GERMAN, {})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t", want_lang="de")
    assert out == GERMAN
    t0, _ = RETRY_TEMPERATURES[0]
    assert vox.calls == [(0.0, 1.0), (t0, 1.0)]


def test_a_window_too_short_to_split_still_gets_every_temperature():
    """A window under two split halves (every pass of the 24B build on a 24 GB
    Mac is ~71 s) stopped after the gentle retry, where the old ladder still
    tried the next temperature -- the rung left to change the decode there."""
    vox = _ScriptedVox([(ENGLISH, {}), (ENGLISH, {}), (GERMAN, {})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t", want_lang="de")
    assert out == GERMAN
    assert [t for t, _ in vox.calls] == [0.0] + [t for t, _ in RETRY_TEMPERATURES]

    vox = _ScriptedVox([(ENGLISH, {})] * 3)          # really English: no penalty rung
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t", want_lang="de")
    assert out == ENGLISH
    assert all(penalty == 1.0 for _, penalty in vox.calls) and len(vox.calls) == 3


def test_a_loop_then_another_language_keeps_climbing():
    """Only a pair of loop-free decodes stops the ladder early. When the greedy
    pass looped, the next rungs still run as before."""
    vox = _ScriptedVox([(DEGEN, {}), (ENGLISH, {}), (GERMAN, {})])
    out = _transcribe_guarded(vox, NO_SPLIT_AUDIO, "de", None, "t", want_lang="de")
    assert out == GERMAN
    assert len(vox.calls) == 3
