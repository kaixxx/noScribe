"""Tests for putting back the opening a long Voxtral pass dropped.

A long window sometimes returns without its first seconds of speech: no loop,
no wrong language, nothing else to notice it by -- the words are just gone. The
repair decodes a short head of the same audio (short windows keep the opening)
and splices back whatever the pass is missing, finding the seam in the text
rather than assuming where it is.

The numbers below are the real ones: on a 1226 s recording whose first 2.4 s
are a remark before the take, the pass came back missing exactly those 12 words.
"""
import numpy as np
import pytest

from noScribe.voxtral_engine import (
    HEAD_ANCHOR_WORDS,
    HEAD_PROBE_MAX_SEC,
    HEAD_PROBE_SEC,
    SAMPLE_RATE,
    _find_anchor,
    _norm_word,
    _recover_lost_head,
)

LEAD = "Zeichnung an, aber ich schlage ja den vorderen Teil eh weg. Ja."
BODY = ("Herzlich willkommen. Willst du Spaß haben und erfolgreich sein in der "
        "digitalen Welt? Ja, unbedingt. Und machst du dir Sorgen über die "
        "Zukunft in der digitalen Welt? Na, ich kenne viele, die sich Sorgen "
        "machen. Kennst du viele, die sich Sorgen machen?")


class FakeVox:
    """Stands in for the model: returns canned probe text, records the windows."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.windows_sec = []

    def transcribe_array(self, audio, language, max_new_tokens=4096, **kw):
        self.windows_sec.append(len(audio) / SAMPLE_RATE)
        return self.replies[min(len(self.windows_sec) - 1, len(self.replies) - 1)]


def _audio(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _log(level, msg):
    pass


# --------------------------------------------------------------------------- #
# _find_anchor
# --------------------------------------------------------------------------- #
def _n(text):
    return [_norm_word(w) for w in text.split()]


def test_anchor_is_found_after_the_dropped_opening():
    at = _find_anchor(_n(f"{LEAD} {BODY}"), _n(BODY)[:HEAD_ANCHOR_WORDS])
    assert at == len(LEAD.split())


def test_anchor_at_zero_when_nothing_was_dropped():
    assert _find_anchor(_n(BODY), _n(BODY)[:HEAD_ANCHOR_WORDS]) == 0


def test_anchor_survives_wording_differences_between_two_decodes():
    """The probe and the pass are independent decodes: "Ja, herzlich" against
    "Ja. Herzlich", "Spaß" against "Spass". Six of eight words still match."""
    probe = _n(f"{LEAD} Herzlich willkommen. Willst du Spass haben und "
               f"erfolgreicher sein in der digitalen Welt?")
    assert _find_anchor(probe, _n(BODY)[:HEAD_ANCHOR_WORDS]) == len(LEAD.split())


def test_no_anchor_when_the_texts_are_unrelated():
    probe = _n("Völlig anderer Text der nichts mit dem Durchgang zu tun hat und "
               "auch sonst keine Wörter teilt")
    assert _find_anchor(probe, _n(BODY)[:HEAD_ANCHOR_WORDS]) is None


def test_the_earliest_of_equally_good_positions_wins():
    """Recovering too little leaves the status quo; too much would duplicate."""
    anchor = _n("eins zwei drei vier")
    probe = _n("eins zwei drei vier fünf eins zwei drei vier")
    assert _find_anchor(probe, anchor, min_hits=4) == 0


# --------------------------------------------------------------------------- #
# _recover_lost_head
# --------------------------------------------------------------------------- #
def test_the_dropped_opening_is_put_back():
    vox = FakeVox(f"{LEAD} {BODY}")
    out = _recover_lost_head(vox, _audio(1226), None, BODY, _log, "Chunk 1/1")
    assert out == f"{LEAD} {BODY}"
    assert vox.windows_sec == [HEAD_PROBE_SEC]


def test_a_pass_that_lost_nothing_is_returned_untouched():
    vox = FakeVox(BODY)
    assert _recover_lost_head(vox, _audio(1226), None, BODY, _log, "x") is BODY


def test_a_short_pass_is_not_probed_at_all():
    """Short windows do not have the defect -- probing them only costs a decode."""
    vox = FakeVox(BODY)
    short = _audio(HEAD_PROBE_SEC * 1.5)
    assert _recover_lost_head(vox, short, None, BODY, _log, "x") is BODY
    assert vox.windows_sec == []


def test_too_little_text_to_anchor_on_is_left_alone():
    vox = FakeVox(BODY)
    stub = " ".join(BODY.split()[:HEAD_ANCHOR_WORDS - 1])
    assert _recover_lost_head(vox, _audio(1226), None, stub, _log, "x") is stub
    assert vox.windows_sec == []


def test_a_degenerate_probe_is_not_spliced_in():
    """A looping probe must never reach the transcript -- and must not be grown.

    A loop is a property of this speech, not of the window length, and a
    degenerate decode is the one that runs to its full token budget. Growing it
    buys a longer loop, not an answer."""
    vox = FakeVox("ja ja " * 60)
    out = _recover_lost_head(vox, _audio(1226), None, BODY, _log, "x")
    assert out is BODY
    assert vox.windows_sec == [HEAD_PROBE_SEC], "grew the probe instead of giving up"


def test_the_probe_grows_when_the_seam_is_beyond_it():
    """A pass that lost more than the first probe covers is still recovered."""
    long_lead = " ".join(f"wort{i}" for i in range(200))
    vox = FakeVox("nichts davon hier drin", f"{long_lead} {BODY}")
    out = _recover_lost_head(vox, _audio(1226), None, BODY, _log, "x")
    assert out == f"{long_lead} {BODY}"
    assert vox.windows_sec == [HEAD_PROBE_SEC, HEAD_PROBE_SEC * 2]


def test_growing_stops_and_the_pass_is_kept():
    vox = FakeVox("nichts davon hier drin")
    out = _recover_lost_head(vox, _audio(1226), None, BODY, _log, "x")
    assert out is BODY
    assert max(vox.windows_sec) <= HEAD_PROBE_MAX_SEC


def test_growing_never_probes_more_than_half_the_pass():
    """The probe is a cheap check, not a second transcription of the window."""
    vox = FakeVox("nichts davon hier drin")
    _recover_lost_head(vox, _audio(200), None, BODY, _log, "x")
    assert max(vox.windows_sec) <= 100


def test_the_first_probe_also_stays_within_half_the_pass():
    """Passes of 90-120 s got the full 60 s first probe -- more than half the
    pass, which the rule above forbids for every probe, not just the grown ones."""
    vox = FakeVox("nichts davon hier drin")
    _recover_lost_head(vox, _audio(100), None, BODY, _log, "x")
    assert vox.windows_sec == [50]


def test_growing_stops_at_the_last_doubling_within_half_the_pass():
    """Stretching the last probe to half the pass decoded the same opening once
    more: 60, 120 and then 200 s on a 400 s pass, and on a real interview a
    226 s probe after 120 s that still missed the anchor."""
    for seconds in (241, 400, 450):
        vox = FakeVox("nichts davon hier drin")
        _recover_lost_head(vox, _audio(seconds), None, BODY, _log, "x")
        assert vox.windows_sec == [HEAD_PROBE_SEC, HEAD_PROBE_SEC * 2], seconds


@pytest.mark.parametrize("lang", [None, "de"])
def test_the_probe_runs_in_the_pass_language(lang):
    """Whatever the pass was decoded with, the probe has to match it."""
    seen = []

    class Recorder(FakeVox):
        def transcribe_array(self, audio, language, max_new_tokens=4096, **kw):
            seen.append(language)
            return super().transcribe_array(audio, language, max_new_tokens, **kw)

    _recover_lost_head(Recorder(f"{LEAD} {BODY}"), _audio(1226), lang, BODY,
                       _log, "x")
    assert seen == [lang]
