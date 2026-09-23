"""Tests for text-based aligner selection (_detect_language / _AlignerPool).

Auto jobs used to fall back to the romanised multilingual aligner even for
plain German audio -- measurably the weakest choice for German word
boundaries. The pool reads the language off each chunk's transcribed text
(which exists before that chunk is aligned) and picks the char-native
per-language model; mixed text without a dominant language keeps the
multilingual fallback, and an explicit-but-wrong language setting warns.
"""
import pytest

import noScribe.voxtral_engine as v
from noScribe.voxtral_engine import (
    ALIGN_MODELS,
    ALIGN_MODEL_MULTILINGUAL,
    _AlignerPool,
    _detect_language,
)

GERMAN = ("Und dann haben wir gesagt, dass wir das nicht einfach so machen, "
          "weil die Sache ja auch für die anderen wichtig ist. Aber wenn wir "
          "ehrlich sind, ist das schon ein großer Schritt, und ich glaube, "
          "dass wir jetzt auf einem guten Weg sind. ") * 4
ENGLISH = ("And then you know we just said that this is not what they wanted, "
           "but if you think about it, they would have been fine with the "
           "idea, because it was just like the other things we did. ") * 4
# Denglisch: German matrix with English phrases mixed in -- the German
# function words still dominate, so the German model must win.
DENGLISCH = ("Und dann haben wir das Mindset komplett geändert, you know, "
             "weil die Journey ja auch ein Commitment ist. Aber wenn wir das "
             "nicht committen, dann ist das eben not the end of the world, "
             "und ich glaube, dass wir da jetzt all in gehen sollten. ") * 4
RUSSIAN = "Мы посмотрели на это и решили, что так будет лучше для всех. " * 6
JAPANESE = "それでは、今日はこのテーマについて話しましょう。よろしくお願いします。" * 6


def test_detects_the_major_languages():
    assert _detect_language(GERMAN)[0] == "de"
    assert _detect_language(ENGLISH)[0] == "en"
    assert _detect_language(RUSSIAN)[0] == "ru"
    assert _detect_language(JAPANESE)[0] == "ja"


def test_denglisch_resolves_to_the_majority_language():
    assert _detect_language(DENGLISCH)[0] == "de"


def test_balanced_mix_and_thin_evidence_stay_undetected():
    assert _detect_language(GERMAN[:200] + " " + ENGLISH[:200] +
                            GERMAN[200:400] + ENGLISH[200:400])[0] is None
    assert _detect_language("Hallo und danke.")[0] is None
    assert _detect_language("")[0] is None


# --------------------------------------------------------------------------- #
# _AlignerPool
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_aligner(monkeypatch):
    loads = []

    class _Fake:
        def __init__(self, model):
            loads.append(model)
            self.model = model

    monkeypatch.setattr(v, "_Aligner", _Fake)
    monkeypatch.setattr(v, "_unfetchable", lambda model: False)  # no network in tests
    return loads


def test_auto_picks_the_char_native_model(fake_aligner):
    logs = []
    pool = _AlignerPool(None, lambda lvl, m: logs.append((lvl, m)))
    a = pool.aligner_for(GERMAN)
    assert a.model == ALIGN_MODELS["de"]
    assert fake_aligner == [ALIGN_MODELS["de"]]
    assert any("Detected language 'de'" in m for _, m in logs)


def test_auto_without_dominant_language_uses_multilingual(fake_aligner):
    pool = _AlignerPool(None, None)
    a = pool.aligner_for("Hallo und danke.")   # thin evidence
    assert a.model == ALIGN_MODEL_MULTILINGUAL


def test_language_change_between_chunks_switches_and_caches(fake_aligner):
    pool = _AlignerPool(None, None)
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]
    assert pool.aligner_for(ENGLISH).model == ALIGN_MODELS["en"]
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]  # cached
    assert fake_aligner == [ALIGN_MODELS["de"], ALIGN_MODELS["en"]]  # 2 loads only


def test_cache_is_bounded(fake_aligner):
    pool = _AlignerPool(None, None)
    pool.aligner_for(GERMAN)
    pool.aligner_for(ENGLISH)
    pool.aligner_for(RUSSIAN)
    assert len(pool._cache) == _AlignerPool.MAX_CACHED


def test_undetected_chunk_keeps_the_previous_choice(fake_aligner):
    pool = _AlignerPool(None, None)
    pool.aligner_for(GERMAN)
    a = pool.aligner_for("Hm. Ja. Okay.")      # no evidence -> stick with de
    assert a.model == ALIGN_MODELS["de"]
    assert fake_aligner == [ALIGN_MODELS["de"]]


def test_explicit_language_stays_but_mismatch_warns_once(fake_aligner):
    logs = []
    pool = _AlignerPool("en", lambda lvl, m: logs.append((lvl, m)))
    a = pool.aligner_for(GERMAN)               # user picked English, audio is German
    assert a.model == ALIGN_MODELS["en"]       # explicit choice is respected
    warns = [m for lvl, m in logs if lvl == "warn"]
    assert len(warns) == 1 and "looks like 'de'" in warns[0]
    pool.aligner_for(GERMAN)                   # second chunk: no repeat warning
    assert len([m for lvl, m in logs if lvl == "warn"]) == 1


def test_explicit_matching_language_does_not_warn(fake_aligner):
    logs = []
    pool = _AlignerPool("de", lambda lvl, m: logs.append((lvl, m)))
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]
    assert not [m for lvl, m in logs if lvl == "warn"]


def test_missing_letters_use_the_spelling_the_model_was_trained_on():
    """Letters outside an aligner's vocabulary are respelled, not dropped.

    Dropping them silently cost real accuracy: the German model has no sharp s,
    so a word ending in one had its closing /s/ unaccounted for and ended up to
    160 ms too early. The models say which spelling to use -- decoded freely the
    German aligner writes "weiss", and the romanised multilingual one writes
    "naechste" -- so the sharp s becomes "ss" and the umlauts become "ae"/"oe"/
    "ue". Substitution only ever applies where the letter is genuinely missing:
    the German model carries the umlauts and must keep them.
    """
    class _FakeAligner:
        _tokenize = v._Aligner._tokenize

        def __init__(self, chars):
            self.vocab = {c: i for i, c in enumerate(chars, start=1)}
            self.delim = None

    german = _FakeAligner("abcdefghijklmnopqrstuvwxyzäöü")
    tokens, _ = german._tokenize(["weiß"])
    assert [c for c in "weiss"] == [
        k for t in tokens for k, i in german.vocab.items() if i == t
    ]
    # the umlauts it does carry stay themselves
    tokens, _ = german._tokenize(["über"])
    assert german.vocab["ü"] == tokens[0]

    mms = _FakeAligner("abcdefghijklmnopqrstuvwxyz'")
    tokens, tok_word = mms._tokenize(["über"])
    assert [c for c in "ueber"] == [
        k for t in tokens for k, i in mms.vocab.items() if i == t
    ]
    # every token still points at its own word
    assert tok_word == [0] * len(tokens)

    # a letter with no mapping is still dropped rather than guessed at
    assert mms._tokenize(["café"])[0] == [mms.vocab[c] for c in "caf"]


def test_unspellable_characters_become_a_wildcard_not_a_hole():
    """A word the model has no letters for still gets tokens of its own.

    Digits are the common case: "20 Sitzungen a 50 Minuten" gave the aligner
    nothing to place, so `_stamps_from_spans` interpolated those words across
    the gap between their neighbours and they came out too wide, carrying the
    prob 0.0 that means "not actually aligned". With the wildcard the DP places
    them on the audio -- measured identical, to the millisecond, to spelling the
    numbers out in German, and 80 ms from the interpolation on average.
    """
    class _FakeAligner:
        _tokenize = v._Aligner._tokenize

        def __init__(self):
            self.vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", 1)}
            self.delim = None
            self.wild = 99

    al = _FakeAligner()
    # off by default, so the prefix-salvage path keeps its unambiguous star
    assert al._tokenize(["20"])[0] == []
    tokens, tok_word = al._tokenize(["20"], wildcard=True)
    assert tokens == [99, 99] and tok_word == [0, 0]
    # punctuation is not audible and must not claim frames
    assert al._tokenize(["!?"], wildcard=True)[0] == []
    # a model without the extra column never emits one
    al.wild = None
    assert al._tokenize(["20"], wildcard=True)[0] == []


def test_an_unusable_respelling_still_reaches_the_wildcard():
    """A mapping that exists but cannot be spelled must not swallow the letter.

    `_tokenize` used to treat "a fallback entry exists" as "the fallback
    worked", so a vocabulary missing one of the replacement letters dropped the
    character silently -- the word lost an anchor and fell back to interpolation
    with nothing in the log. Both shipped models carry a-z so no current entry
    hits this, which is exactly why it needs a test.
    """
    class _FakeAligner:
        _tokenize = v._Aligner._tokenize

        def __init__(self, chars):
            self.vocab = {c: i for i, c in enumerate(chars, start=1)}
            self.delim = None
            self.wild = 99

    without_s = _FakeAligner("abcdefghijklmnopqrtuvwxyz")   # no "s" for "ss"
    assert without_s._tokenize(["weiß"], wildcard=True)[0][-1] == 99
    with_s = _FakeAligner("abcdefghijklmnopqrstuvwxyz")
    assert 99 not in with_s._tokenize(["weiß"], wildcard=True)[0]
    # without the wildcard the old behaviour stands: the letter is dropped
    assert without_s._tokenize(["weiß"])[0] == [
        without_s.vocab[c] for c in "wei"
    ]


def test_a_language_the_detector_cannot_name_draws_no_warning(fake_aligner):
    """All Cyrillic reads as Russian, so a correctly pinned Ukrainian file was
    told "set to 'uk' but the transcript looks like 'ru' ... check the language
    setting" -- the one advice that is wrong for it."""
    ukrainian = ("Ми ще раз усе обговорили, бо я не був певен, чи це підходить, "
                 "але тепер усе зрозуміло, і ми вже рухаємося далі, бо коли інші "
                 "ще раз подивляться, у мене є відчуття, що все буде добре.")
    logs = []
    pool = _AlignerPool("uk", lambda lvl, m: logs.append((lvl, m)))
    pool.aligner_for(ukrainian)
    assert not [m for lvl, m in logs if lvl == "warn"]
