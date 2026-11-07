"""Where a chunk is cut, and how long a subtitle cue may become.

Both are places where a wrong value produces no visible error, only a silently
worse transcript: a cut in the middle of a word, or a cue that stays on screen
for half a minute. That is why they are pinned here.
"""
import numpy as np

from noScribe.voxtral_engine import (
    QUIET_LEVEL,
    SAMPLE_RATE,
    SUB_MAX_SEC,
    _chunk_boundaries,
    _pass_bounds,
    _frame_energy,
    _segments_from_words,
)


def _noise(rng, n, amp):
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _speech(rng, seconds, amp=0.30):
    return _noise(rng, int(seconds * SAMPLE_RATE), amp)


def _fill(arr, t0, t1, amp, rng):
    i0, i1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
    arr[i0:i1] = _noise(rng, i1 - i0, amp)


# --------------------------------------------------------------------------- #
# Cut search
# --------------------------------------------------------------------------- #
def test_quiet_speech_is_not_mistaken_for_a_pause():
    """_frame_energy yields power (amplitude^2). With QUIET_LEVEL applied to it
    unsquared, the threshold sat at -8.2 dB instead of -16.5 dB -- that is, at
    quiet speech, not at a pause. Every 400 ms passage of unstressed syllables
    thus counted as a "clear speaker pause" and, being closer to the target,
    beat the real pause: the cut landed in the middle of a word."""
    rng = np.random.default_rng(0)
    audio = _speech(rng, 1200)
    _fill(audio, 589.0, 590.5, 0.0002, rng)                  # real pause
    _fill(audio, 598.2, 598.8, 0.30 * 10 ** (-12 / 20), rng)  # quiet speech

    bounds = _chunk_boundaries(audio, 600 * SAMPLE_RATE, 90 * SAMPLE_RATE,
                               20 * SAMPLE_RATE, 600 * SAMPLE_RATE)
    cut = bounds[1] / SAMPLE_RATE
    assert 589.0 <= cut <= 590.5, f"cut at {cut:.2f}s instead of in the pause"


def test_noisy_recording_still_finds_its_best_pause():
    """Counter-check for the sharper threshold: a recording with a high noise
    floor (room tone, hum) has no real silence anywhere. It must therefore not
    fall back to the unsnapped target value but still take the relative
    minimum."""
    rng = np.random.default_rng(1)
    audio = _speech(rng, 1200)
    _fill(audio, 589.0, 590.5, 0.30 * 10 ** (-10 / 20), rng)   # only a dip

    bounds = _chunk_boundaries(audio, 600 * SAMPLE_RATE, 90 * SAMPLE_RATE,
                               20 * SAMPLE_RATE, 600 * SAMPLE_RATE)
    cut = bounds[1] / SAMPLE_RATE
    assert 589.0 <= cut <= 590.5, f"cut at {cut:.2f}s, dip missed"


def test_a_backward_snap_does_not_leave_a_sub_second_tail():
    """The first cut snapped back to a pause at 589.2 s, leaving 600.5 s --
    just over the 600 s cap, so one more cut was forced. It aimed at a full
    pass, found the pause at 1189.3 s and left a final pass of 0.55 s: a whole
    Voxtral decode (and an aligner call) for half a second of audio, with no
    context, although the docstring promises tails merge instead."""
    rng = np.random.default_rng(3)
    audio = _speech(rng, 1190)
    for t in (589.2, 1189.3):
        _fill(audio, t, t + 0.6, 0.0002, rng)
    passes = np.diff(_pass_bounds(audio, 600)) / SAMPLE_RATE
    assert passes.max() <= 600, passes
    assert passes.min() >= 600 / 8, passes


def test_pass_plan_invariants_on_random_files():
    """Passes never exceed the RAM-safe length (memory safety), never run
    backward, and never come out as a tiny leftover -- whatever the length of
    the file and wherever its pauses happen to sit. Short passes keep the audio
    small; the invariants do not depend on the scale."""
    rng = np.random.default_rng(7)
    base = _speech(rng, 420)
    for _ in range(60):
        chunk_sec = float(rng.uniform(60, 120))
        n = int(rng.uniform(5, 420) * SAMPLE_RATE)
        audio = base[:n].copy()
        for t in rng.uniform(0, n / SAMPLE_RATE, size=int(rng.integers(0, 12))):
            _fill(audio, t, min(t + rng.uniform(0.2, 1.5), n / SAMPLE_RATE),
                  0.0002, rng)
        bounds = _pass_bounds(audio, chunk_sec)
        passes = np.diff(bounds)
        max_len = int(chunk_sec * SAMPLE_RATE)
        assert bounds[0] == 0 and bounds[-1] == n
        assert (passes > 0).all(), bounds
        assert passes.max() <= max_len, (chunk_sec, n, bounds)
        if n > max_len:
            assert passes.min() >= max_len // 8, (chunk_sec, n, bounds)


def test_quiet_level_is_compared_in_the_power_domain():
    """The constant is an amplitude ratio; _frame_energy is power. The
    difference is 2x in dB and exactly the mistake fixed above."""
    rng = np.random.default_rng(2)
    loud = _frame_energy(_speech(rng, 1.0, 0.30), 800)
    quiet = _frame_energy(_speech(rng, 1.0, 0.30 * QUIET_LEVEL), 800)
    # The quiet passage sits at QUIET_LEVEL**2 of the power, not at QUIET_LEVEL.
    assert np.median(quiet) < np.median(loud) * QUIET_LEVEL ** 2 * 2
    assert np.median(quiet) > np.median(loud) * QUIET_LEVEL ** 2 * 0.5


# --------------------------------------------------------------------------- #
# Cue length
# --------------------------------------------------------------------------- #
def _cues(stamps):
    return [(round(s["start"], 2), round(s["end"], 2), s["text"].strip())
            for s in _segments_from_words(stamps)]


def _w(word, start, end, prob=0.9):
    return {"word": word, "start": start, "end": end, "prob": prob}


def test_a_long_gap_does_not_produce_an_overlong_cue():
    """SUB_MAX_SEC was only checked once the word was already in the cue, and
    flush() can only end a cue including that word -- so every speech pause
    longer than the ceiling produced a cue of exactly that length (measured:
    31 s for two words around a piece of music)."""
    cues = _cues([_w("Musik", 0.50, 1.00), _w("beginnt.", 31.00, 31.50)])
    assert len(cues) == 2, cues
    assert all(end - start <= SUB_MAX_SEC for start, end, _ in cues), cues


def test_an_ordinary_thinking_pause_also_splits_the_cue():
    """No special case for music: a 9-second thinking pause in the middle of a
    sentence was already enough for a 10.4 s cue."""
    cues = _cues([_w("Ich", 0.0, 0.3), _w("glaube", 0.4, 0.9),
                  _w("dass", 10.0, 10.4), _w("es.", 10.5, 10.8)])
    assert len(cues) == 2, cues
    assert all(end - start <= SUB_MAX_SEC for start, end, _ in cues), cues


def test_normal_speech_still_lands_in_one_cue():
    """Counter-check: without a large gap the new pre-check must not chop anything up."""
    stamps = [_w(w, i * 0.4, i * 0.4 + 0.35)
              for i, w in enumerate("wir haben das gestern kurz besprochen".split())]
    assert len(_cues(stamps)) == 1


def test_no_cue_has_zero_duration():
    """A cue with start == end is invalid per the WebVTT specification; the
    writer in utils does not check for it, so it has to be right here."""
    stamps = [_w("Das", 0.0, 0.3), _w("war", 0.4, 0.7),
              _w("1990.", 0.7, 0.7, prob=0.0),      # OOV, interpolated
              _w("Genau.", 3.0, 3.4)]
    for start, end, text in _cues(stamps):
        assert end > start, f"cue without duration: {text!r}"


def test_a_lone_word_never_keeps_its_own_cue():
    """The repair step against one-word fragments applied the duration
    condition to a single word as well. Measured on "Digitale Welt Video 7"
    (00:03:47): SUB_MAX_CHARS closed the cue after "...Reorganisation" (49
    characters), and "angesagt." ran, stretched, ~1.2 s into the pause before
    the next cue -- the condition flipped, and the word stayed a segment of
    its own. A segment that short takes its speaker from whatever the
    diarization puts underneath it: 0.4 s of a backchannel cluster gave the
    word in the middle of the sentence its own speaker and its own paragraph."""
    words = "haben sie jetzt schon die nächste Reorganisation".split()
    stamps = [_w(w, 225.0 + i * 0.40, 225.0 + i * 0.40 + 0.35)
              for i, w in enumerate(words)]
    stamps.append(_w("angesagt.", 227.85, 229.05))   # 1.2 s, stretched
    cues = _cues(stamps)
    assert len(cues) == 1, cues
    assert cues[0][2].endswith("Reorganisation angesagt."), cues


def test_a_lone_word_after_a_long_pause_still_stands_alone():
    """Counter-check: the length limits stay in charge. If the word lies so far
    behind the previous cue that the merged cue would reach beyond
    SUB_MAX_SEC + 2, it must not be created -- otherwise a subtitle stands
    across the whole pause."""
    stamps = [_w("Also", 0.0, 0.4), _w("gut", 0.5, 0.9)]
    stamps.append(_w("weiter.", 12.0, 13.2))
    cues = _cues(stamps)
    assert len(cues) == 2, cues


def test_the_two_word_fragment_keeps_its_duration_guard():
    """Only the single word is independent of its duration. Two words that
    together last longer than 1.2 s are an ordinary short cue and are still
    not glued on."""
    stamps = [_w("Das", 0.0, 0.3), _w("ist", 0.4, 0.7), _w("so.", 0.8, 1.1)]
    stamps.append(_w("Sehr", 4.0, 4.9))
    stamps.append(_w("gut.", 5.0, 5.8))               # 2 words, 1.8 s
    cues = _cues(stamps)
    assert len(cues) == 2, cues
