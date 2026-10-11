"""
Tests for the `transcript_corrections.py` module.
"""

import pytest

from noScribe import transcript_corrections as tc


def test_apply_corrections_is_literal():
    """A user-supplied replacement must never be read as a regex template."""
    rules = [(__import__("re").compile(r"(?<!\w)foobar(?!\w)", 2), r"Foo\Bar\1")]
    assert tc.apply_corrections("say foobar now", rules) == r"say Foo\Bar\1 now"


def test_load_corrections_skips_malformed(tmp_path, caplog):
    """Malformed entries are reported, not silently dropped."""
    f = tmp_path / "corrections.yml"
    f.write_text(
        "- to: VitaFlor\n"
        "  from: [vitaflor, 'vita flor']\n"
        "- 17\n"                       # not a mapping
        "- to: OnlyTo\n",              # missing 'from'
        encoding="utf-8",
    )
    rules = tc.load_corrections(str(f))
    assert len(rules) == 1
    assert tc.apply_corrections("ich nehme vita flor", rules) == "ich nehme VitaFlor"


def test_load_corrections_missing_file():
    assert tc.load_corrections(None) == []
    assert tc.load_corrections("/nonexistent/corrections.yml") == []


def test_load_corrections_unparsable_file_is_reported_not_raised(tmp_path, caplog):
    """A hand-edited file with a YAML error must not crash the job: the loader
    warns and the run continues without corrections."""
    f = tmp_path / "corrections.yml"
    f.write_text("- to: [unclosed\n", encoding="utf-8")
    assert tc.load_corrections(str(f)) == []
    assert "Could not read corrections file" in caplog.text


def test_load_corrections_rejects_a_non_list_root(tmp_path, caplog):
    """The format is a list of entries; a bare mapping is the likeliest typo."""
    f = tmp_path / "corrections.yml"
    f.write_text("to: VitaFlor\nfrom: vitaflor\n", encoding="utf-8")
    assert tc.load_corrections(str(f)) == []
    assert "must contain a list" in caplog.text


def test_longest_pattern_wins_within_an_entry(tmp_path):
    """Patterns are tried longest first, so a phrase is not eaten by its own
    prefix: "vita flor oil" must not become "VitaFlor oil"."""
    f = tmp_path / "corrections.yml"
    f.write_text(
        "- to: VitaFlor-Oil\n"
        "  from: ['vita flor', 'vita flor oil']\n",
        encoding="utf-8",
    )
    rules = tc.load_corrections(str(f))
    assert tc.apply_corrections("take vita flor oil daily", rules) == "take VitaFlor-Oil daily"


def test_default_template_has_no_active_rules(tmp_path):
    """The shipped file is a commented template: it must not rewrite anything."""
    path = tc.ensure_default_file(str(tmp_path))
    assert tc.load_corrections(path) == []


def test_only_spellings_that_sound_the_same_share_a_key():
    """What a listener cannot tell apart lines up; any change of sound does
    not -- a looser rule made other people's names and fillers into speakers'
    names (see _SAME_SOUND)."""
    def same(word, name):
        return tc._spelling_key(word) == tc._spelling_key(name)
    assert same("Mohna", "Mona")                 # a silent h
    assert same("Noa", "Noah")
    assert same("Marcus", "Markus")              # c spoken as k
    assert same("Schmidt", "Schmit")
    assert same("Steffy", "Steffi")              # a final y
    assert not same("Sybille", "Sibille")        # but inside a name y can be ü
    assert not same("Johanna", "Joanna")         # an h that is spoken
    assert not same("Muna", "Mona")              # every vowel counts
    assert not same("Leni", "Lena")
    assert not same("\u00c4hm", "Ann")
    assert not same("Hamma", "Hanna")            # m is not n
    assert not same("Ole", "\u00d6le")          # nor is an umlaut nothing
    assert not same("Lucie", "Luzie")            # c before i is not z
    assert not same("Ela", "Ella")               # a doubled consonant shortens the vowel
    assert not same("Joon", "John")              # and a doubled vowel is another one
    assert not same("Yvonne", "Ivonne")          # a leading y is left as it is
    assert same("Rafael", "Raphael")


# A stand-in for the macOS dictionaries, so these tests run where there are
# none (Linux CI). Every word the tests use that is not a mis-hearing is in it.
_WORDS = {
    "de": {"Ich", "Muster", "Und", "Heute", "Mit", "Hier", "Hallo", "Sagt",
           "Mona", "Markus", "Marcus", "Der", "Mann", "Kam", "Seiner", "Mama",
           "Dann", "Wir", "Fahren", "Nach", "Kerl", "Karl", "Rose", "Rosa",
           "Steffi", "Mohn", "Raphael"},
    "en": {"Then", "Said", "Mona", "Moan", "Mane", "Rome", "Lena", "Lina",
           "Hello", "Were", "Warren"},
    "xx": set(),        # a dictionary that knows none of the words
}


@pytest.fixture
def dictionaries(monkeypatch):
    monkeypatch.setattr(tc, "_dictionary_language",
                        lambda language: language if language in _WORDS else None)
    monkeypatch.setattr(tc, "_is_word", lambda word, language: word in _WORDS[language])


def test_apply_name_corrections_fixes_spelling(dictionaries):
    """A spelling of a name that is no word takes the one the user provided."""
    names = ["Mona", "Steffi"]
    out = tc.apply_name_corrections("Ich bin Mohna Muster, und Steffy auch.", names, "de")
    assert out == "Ich bin Mona Muster, und Steffi auch."
    # already correct -> untouched
    assert tc.apply_name_corrections("Hallo Steffi, sagt Mona.", names, "de") == "Hallo Steffi, sagt Mona."
    # lower case is no name ("mohna" mid-sentence), and the curly possessive is
    # a possessive like the straight one
    assert tc.apply_name_corrections("die mohna und Mohna\u2019s Hut.", names, "de") \
        == "die mohna und Mohna\u2019s Hut."


def test_a_mis_hearing_that_changes_a_sound_is_left_alone(dictionaries):
    """"Muna" is no word either, but it is not how "Mona" sounds: it may be
    someone else, and "Leni" next to a speaker called Lena certainly is."""
    assert tc.apply_name_corrections("Ich bin Muna.", ["Mona"], "de") == "Ich bin Muna."
    assert tc.apply_name_corrections("Then Leni said hello.", ["Lena"], "en") \
        == "Then Leni said hello."


def test_apply_name_corrections_works_in_any_language_with_a_dictionary(dictionaries):
    assert tc.apply_name_corrections("Then Mohna said hello.", ["Mona"], "en") \
        == "Then Mona said hello."


def test_apply_name_corrections_takes_each_capitalised_part_of_a_full_name(dictionaries):
    """A speaker entered as "Mona Muster" or "Lena-Mona" is two names; the old
    `isalpha()` filter dropped such an entry whole. A particle is no name: "del"
    in "Ana del R\u00edo", and a name typed in lower case would write itself
    in lower case into the transcript ("Mohna" -> "mona")."""
    assert tc.apply_name_corrections("Ich bin Mohna Muhster.", ["Mona Muster"], "de") \
        == "Ich bin Mona Muster."
    assert tc.apply_name_corrections("Ich bin Mohna.", ["Lena-Mona"], "de") == "Ich bin Mona."
    assert tc.apply_name_corrections("Ich bin Mohna.", ["mona"], "de") == "Ich bin Mohna."


def test_apply_name_corrections_keeps_a_real_name_that_sounds_alike(dictionaries):
    """"Marcus" is a name of its own and may be someone else; only a spelling
    that is no word at all is taken for a mis-hearing."""
    assert tc.apply_name_corrections("Heute mit Marcus hier.", ["Markus"], "de") \
        == "Heute mit Marcus hier."


def test_apply_name_corrections_leaves_an_ambiguous_word_alone(dictionaries):
    """"Sahra" is spelled like both speakers; picking one would be a guess."""
    assert tc.apply_name_corrections("Ich bin Sahra.", ["Sarah", "Sara"], "de") \
        == "Ich bin Sahra."


def test_apply_name_corrections_leaves_real_words_alone(dictionaries):
    """Every German noun is capitalised, and "Mohn" is spelled like a speaker
    called Mon; the dictionary keeps it. So does a possessive, with a straight
    or a curly apostrophe: its stem is not a word of its own."""
    for sentence, names in [
        ("Dann Mohn und Marcus.", ["Mon", "Markus"]),
        ("Mohna's Hut und Mohna\u2019s Tasche.", ["Mona"]),
    ]:
        assert tc.apply_name_corrections(sentence, names, "de") == sentence


def test_a_word_any_of_the_languages_knows_is_kept(dictionaries):
    """Text the script check reads as one language may be in another, and a
    dictionary knows nothing of its neighbour's words: a word stays when any
    of the languages the text may be in knows it."""
    assert tc.apply_name_corrections("Mohn kam.", ["Mon"], ["xx", "de"]) == "Mohn kam."
    assert tc.apply_name_corrections("Mohna kam.", ["Mona"], ["xx", "de"]) == "Mona kam."


def test_apply_name_corrections_needs_a_dictionary_for_the_language(dictionaries):
    """Without one there is no telling a mis-heard name from a word -- and
    macOS, asked about a language it has no dictionary for, calls every word
    correct -- so nothing is changed."""
    for language in ("zh", None, "", "auto", ["zh", None]):
        assert tc.apply_name_corrections("Mohna kam.", ["Mona"], language) == "Mohna kam."


def test_dictionary_language_maps_codes_to_what_the_system_has():
    pytest.importorskip("AppKit")
    if tc._spell_checker() is None:
        pytest.skip("no spell checker")
    available = [str(lang) for lang in tc._spell_checker().availableLanguages()]
    if "pt" not in available and any(l.startswith("pt_") for l in available):
        assert tc._dictionary_language("pt").startswith("pt_")
    assert tc._dictionary_language("xx") is None
    assert tc._dictionary_language(None) is None
    assert tc._dictionary_language("auto") is None


def test_apply_name_corrections_with_the_macos_dictionary():
    """The same guards against the real dictionaries the engine will ask."""
    pytest.importorskip("AppKit")
    if tc._dictionary_language("de") is None or tc._dictionary_language("en") is None:
        pytest.skip("no German or English dictionary installed")
    for text, names, language in [
        ("Dann Mohn und Marcus.", ["Mon", "Markus"], "de"),   # spelled alike, known
        ("Then Carl said hello.", ["Karl"], "en"),
        ("\u00c4hm, also das war so.", ["Ann"], "de"),       # a filler
        ("Hamma scho, sagt Leni.", ["Hanna", "Lena"], "de"),  # dialect, another name
    ]:
        assert tc.apply_name_corrections(text, names, language) == text
    # Under the load of a full test run the spell server sometimes stalls, and
    # a stalled lookup answers "a word" (see _is_word); asking again is what
    # the next chunk of a real job does too.
    for text, language, expected in [
        ("Ich bin Mohna Muster.", "de", "Ich bin Mona Muster."),
        ("Then Mohna came to Rome.", "en", "Then Mona came to Rome."),
    ]:
        for _ in range(5):
            out = tc.apply_name_corrections(text, ["Mona"], language)
            if out == expected:
                break
        assert out == expected


_NOT_FOUND = 2**63 - 1   # NSNotFound: the location of "no misspelling"


class _FakeChecker:
    """Answers like NSSpellChecker, with every word unknown (flagged whole)
    unless told to answer as a stalled spell server does, or to flag a range
    that is not the word."""
    def __init__(self, flagged=None):
        self.flagged = flagged
        self.asked = []

    def checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
            self, word, start, language, wrap, tag, count):
        from types import SimpleNamespace
        self.asked.append(word)
        location, length = self.flagged or (0, len(word))
        return SimpleNamespace(location=location, length=length), 1


@pytest.mark.parametrize("flagged", [
    (_NOT_FOUND, 0),     # a stalled server: measured to report no misspelling
    (1, 2),              # a range that is not the word
    (0, 2),              # nor is one that covers only its start
])
def test_only_a_clear_unknown_lets_a_name_in(monkeypatch, flagged):
    """A wrong "unknown" would rewrite a real word into a name, so only a
    verdict on the whole word counts -- and a stall, which answers "no
    misspelling", is not remembered: the next chunk asks again."""
    monkeypatch.setattr(tc, "_dictionary_language", lambda language: "de")
    monkeypatch.setattr(tc, "_spell_checker", lambda: _FakeChecker(flagged))
    assert tc.apply_name_corrections("Die Mohna kam.", ["Mona"], "de") == "Die Mohna kam."
    monkeypatch.setattr(tc, "_spell_checker", lambda: _FakeChecker())
    assert tc.apply_name_corrections("Die Mohna kam.", ["Mona"], "de") == "Die Mona kam."


def test_each_word_is_looked_up_once_per_chunk(monkeypatch):
    checker = _FakeChecker()
    monkeypatch.setattr(tc, "_dictionary_language", lambda language: "de")
    monkeypatch.setattr(tc, "_spell_checker", lambda: checker)
    assert tc.apply_name_corrections("Mohna, Mohna und Mohna.", ["Mona"], "de") \
        == "Mona, Mona und Mona."
    assert checker.asked == ["Mohna"]


def test_a_spell_checker_that_fails_costs_no_job(monkeypatch):
    """Name correction runs after each pass is decoded; the dictionary list was
    the one AppKit call outside a guard, and an error there failed the job."""
    class _Broken:
        def availableLanguages(self):
            raise RuntimeError("the spell server is gone")

    monkeypatch.setattr(tc, "_spell_checker", lambda: _Broken())
    tc._dictionary_language.cache_clear()
    try:
        assert tc.apply_name_corrections("Die Mohna kam.", ["Mona"], "de") == "Die Mohna kam."
    finally:
        tc._dictionary_language.cache_clear()


def test_apply_name_corrections_ignores_empty_and_short_names(dictionaries):
    assert tc.apply_name_corrections("Mohna kam.", [], "de") == "Mohna kam."
    assert tc.apply_name_corrections("Mohna kam.", ["Ro"], "de") == "Mohna kam."


def test_degenerate_detector_separates_real_text_from_loops():
    """The repetition-loop detector must catch a runaway pass without ever
    flagging real speech.

    Calibrated on real transcripts (Whisper and Voxtral, German): the longest
    run of identical words is 2 and the compression ratio 2.59-2.63, while an
    observed loop ran to 690 identical words at a ratio of 5.75.
    """
    from noScribe.voxtral_engine import _looks_degenerate

    real = _clean_german_paragraph()
    assert not _looks_degenerate(real)
    # a pass that collapses into a loop, also when it starts out fine
    assert _looks_degenerate("Jetzt. " * 40)
    assert _looks_degenerate(real + " " + "Jetzt. " * 200)
    # short passes are never judged (a brief pass is legitimate)
    assert not _looks_degenerate("Ja ja ja ja ja.")
    assert not _looks_degenerate("")


# Cycles taken from loops that reached finished transcripts before the
# detector counted cycles instead of identical neighbours. Every one of them
# is two words long, so no two adjacent words are equal and the old run
# counter measured 1 where the truth ran from 19 to 68 repeats.
OBSERVED_LOOP_CYCLES = ["dass das, ", "es ist, ", "ja, ich, ", "Macht ist... "]


@pytest.mark.parametrize("cycle", OBSERVED_LOOP_CYCLES)
def test_two_word_loops_are_caught_even_when_buried_in_real_speech(cycle):
    """Regression: each of these shipped in a transcript unflagged.

    The loop is checked in the shape it actually occurred -- a short burst
    inside an otherwise clean chunk. That matters, because the compression
    ratio is computed over the whole pass and cannot see a local loop: at the
    real 4% ratio it moved from 2.6 to 2.8, far below the 4.0 threshold.
    """
    from noScribe.voxtral_engine import _looks_degenerate

    clean = _clean_german_paragraph()
    loop = cycle * 19  # the mildest two-word loop actually observed
    assert _looks_degenerate(loop.strip())
    assert _looks_degenerate(clean + " " + loop)
    # and still when the loop is the same 4% slice of the chunk it was in the
    # transcript that prompted this test
    padded = " ".join([clean] * ((len(loop.split()) * 25) // len(clean.split()) + 1))
    assert _looks_degenerate(padded + " " + loop)


def test_real_speech_stays_clean_under_the_cycle_counter():
    """The counter must not fire on the repetition real speech does contain.

    Over 278 real units (Voxtral chunks and finished Whisper transcripts)
    nothing that reads as speech exceeded 11 repeats, against a threshold of
    12. The margin is thin by measurement, not by choice -- see the constant.
    """
    from noScribe.voxtral_engine import (DEGENERATE_CYCLE_REPEATS,
                                         _longest_cycle_repeats, _looks_degenerate)

    clean = _clean_german_paragraph()
    assert _longest_cycle_repeats(clean.split()) < DEGENERATE_CYCLE_REPEATS
    # Emphatic repetition and a filler stutter, at the length real speech
    # reaches. A longer synthetic case used to live here and was the only
    # reason the threshold sat at 20; the corpus never produced anything like
    # it, while a real 19-repeat loop went undetected because of it.
    for phrase in ("Ja, ja, ja, genau so. ", "Also, also, also, ich meine. "):
        assert not _looks_degenerate(clean + " " + phrase * 2 + clean)


def _clean_german_paragraph():
    return (
        "Und wenn du schon ein Produkt nimmst, hervorragend, dann hast du schon einen "
        "Schritt weiter gemacht als viele andere. Meine Empfehlung wäre trotzdem, es "
        "einmal auszuprobieren. Der Test zeigt dir nämlich ganz konkret, wo deine Werte "
        "wirklich liegen, und danach kannst du die Veränderung tatsächlich messen. Das "
        "Schöne daran ist, dass du überhaupt kein Risiko eingehst. Selbst wenn sich "
        "herausstellt, dass dein bisheriges Öl völlig in Ordnung war, hast du wenigstens "
        "Klarheit gewonnen. Für mich persönlich war genau das der entscheidende Punkt, "
        "weil ich vorher jahrelang im Dunkeln getappt bin. Wer Kinder hat, sollte sie "
        "unbedingt ebenfalls testen lassen, denn die brauchen anteilig sogar mehr."
    )


def _fake_audio(seconds=200):
    import numpy as np
    from noScribe.voxtral_engine import SAMPLE_RATE
    a = (np.random.RandomState(0).randn(seconds * SAMPLE_RATE).astype(np.float32)) * 0.1
    mid = seconds // 2
    a[mid * SAMPLE_RATE:int((mid + 0.5) * SAMPLE_RATE)] = 0.0   # a clear pause to cut at
    return a


VARIED_TEXT = [
    "Ich kann es euch jetzt auch nicht nicht erzählen, weil die Sache zu wichtig ist. "
    "Wir haben lange überlegt und uns dann für den direkten Weg entschieden. "
    "Der Test zeigt dir schwarz auf weiß, wo du stehst, und danach entscheidest du.",
    "Was mich am meisten überzeugt hat, war die Klarheit der Werte nach acht Wochen. "
    "Vorher hätte ich behauptet, alles richtig zu machen, und lag damit daneben. "
    "Wer Kinder hat, sollte sie ebenfalls testen lassen, ihr Bedarf liegt höher.",
]


class _FakeVoxtral:
    """Loops on long passes at penalty 1.0 (any temperature), transcribes
    anything else fine -- exercising the split stage of the ladder."""

    def __init__(self, always_loop=False):
        self.calls = []
        self.always_loop = always_loop
        self._i = 0

    def transcribe_array(self, audio, language, max_new_tokens=0, repetition_penalty=1.0,
                         token_cb=None, temperature=0.0, seed=None, info=None):
        from noScribe.voxtral_engine import SAMPLE_RATE
        dur = len(audio) / SAMPLE_RATE
        self.calls.append((round(dur), repetition_penalty))
        if repetition_penalty == 1.0 and (self.always_loop or dur > 150):
            return "Und das ist jetzt passiert. " + "Jetzt. " * 300
        text = VARIED_TEXT[self._i % len(VARIED_TEXT)]
        self._i += 1
        return text


def test_looping_pass_is_split_not_penalised():
    """Splitting must be tried first: a penalty deletes meaningful repetitions
    (it turned "nicht nicht erzählen" into "nicht erzählen", inverting it)."""
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _FakeVoxtral()
    out = _transcribe_guarded(vox, _fake_audio(), "de", None, "Pass 1/1")
    assert not _looks_degenerate(out)
    assert "nicht nicht" in out                       # meaningful repetition survives
    assert all(p == 1.0 for _, p in vox.calls)        # no penalty was needed
    # greedy 200s, gentle-sampling retry 200s (this fake loops regardless of
    # temperature), then the split resolves it at 2x100s.
    assert [d for d, _ in vox.calls] == [200, 200, 100, 100]


def test_penalty_is_the_last_resort_when_splitting_fails():
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _FakeVoxtral(always_loop=True)
    out = _transcribe_guarded(vox, _fake_audio(), "de", None, "Pass 1/1")
    assert not _looks_degenerate(out)
    assert vox.calls[0] == (200, 1.0)                 # unpenalised attempt first
    penalties = [p for _, p in vox.calls if p != 1.0]
    assert penalties                                  # ...penalty only afterwards
    assert penalties[0] == 1.01                       # and the gentlest one first


def test_split_point_lands_in_the_pause():
    from noScribe.voxtral_engine import _quietest_split, SAMPLE_RATE

    cut = _quietest_split(_fake_audio(200)) / SAMPLE_RATE
    assert 99.9 <= cut <= 100.6


def test_model_ram_hint_round_trips():
    """The picker shows the RAM requirement, but the app must still get the
    plain model name back out of the decorated entry."""
    from types import SimpleNamespace
    import noScribe.main as m
    from noScribe import transcription, voxtral_engine as v

    stub = SimpleNamespace(whisper_models={}, MODEL_LABEL_SEP=m.App.MODEL_LABEL_SEP,
                           _model_label_to_name={})
    stub.whisper_models["voxtral-mini-8bit"] = transcription.WhisperModel(
        name="voxtral-mini-8bit", path=None, engine="voxtral",
        repo="models/voxtral-mini-8bit")
    stub.whisper_models["precise"] = transcription.WhisperModel(name="precise", path=None)

    label = m.App.model_label(stub, "voxtral-mini-8bit")
    assert "GB RAM" in label
    # As the dropdown does, remember the label -> name mapping.
    stub._model_label_to_name = {label: "voxtral-mini-8bit"}
    assert m.App.model_key(stub, label) == "voxtral-mini-8bit"
    # A decorated label not in the map still round-trips via the separator split.
    stub._model_label_to_name = {}
    assert m.App.model_key(stub, label) == "voxtral-mini-8bit"
    # non-Voxtral models are shown unchanged
    assert m.App.model_label(stub, "precise") == "precise"
    assert m.App.model_key(stub, "precise") == "precise"


def test_model_too_large_for_the_machine_is_refused():
    """Starting a run that cannot fit does not fail loudly -- it swaps until
    nothing progresses. Refuse up front instead."""
    import pytest
    from noScribe import voxtral_engine as v

    orig = v._total_ram_gb
    try:
        v._total_ram_gb = lambda: 16.0
        with pytest.raises(MemoryError):
            v._auto_chunk_sec("models/voxtral-small-8bit", None)
        # the small 3B build still works on the same machine
        assert v._auto_chunk_sec("models/voxtral-mini-8bit", None) > 0
    finally:
        v._total_ram_gb = orig


class _StopRun(Exception):
    """Sentinel raised by the stubbed model constructor: proves the run got
    exactly as far as the model load and no further."""


def _tiny_wav(tmp_path):
    import numpy as np
    import soundfile as sf
    p = tmp_path / "t.wav"
    sf.write(p, np.zeros(16000, dtype="float32"), 16000)
    return str(p)


def test_refusal_happens_before_the_model_is_loaded(monkeypatch, tmp_path):
    """The point of refusing an oversized model is to refuse *before* 20+ GB of
    weights push the machine into swap -- so transcribe() must size passes
    first and only then construct the model."""
    import pytest
    from noScribe import voxtral_engine as v

    wav = _tiny_wav(tmp_path)
    loaded = []
    monkeypatch.setattr(v, "_Voxtral", lambda repo: loaded.append(repo))
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 16.0)
    with pytest.raises(MemoryError):
        v.transcribe(wav, voxtral_repo="models/voxtral-small-8bit")
    assert loaded == []  # refused without ever touching the weights
    # ...but a missing FILE is a missing file, not a memory problem
    with pytest.raises(FileNotFoundError):
        v.transcribe(str(tmp_path / "missing.wav"),
                     voxtral_repo="models/voxtral-small-8bit")
    assert loaded == []


def test_pinned_chunk_sec_cannot_bypass_the_memory_ceiling(monkeypatch, tmp_path):
    """A voxtral_chunk_sec pinned while experimenting with mini must not let a
    hungrier model run passes whose working set cannot fit (that run would swap
    forever, not fail) -- nor exceed the model-context cap on a huge machine."""
    import pytest
    from noScribe import voxtral_engine as v

    wav = _tiny_wav(tmp_path)

    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 32.0)
    warnings = []
    with pytest.raises(_StopRun):  # got past sizing, stopped at model load
        v.transcribe(wav, voxtral_repo="models/voxtral-small-6bit",
                     chunk_sec=1500, need_timestamps=False,
                     log_cb=lambda lvl, msg: warnings.append((lvl, msg)))
    # 6-bit small on 32 GB: hard ceiling (32-6-20.9)/0.0135 = 377s
    assert any(lvl == "warn" and "voxtral_chunk_sec" in msg and "377" in msg
               for lvl, msg in warnings)
    # a model that cannot fit at all is refused even with a pinned length
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 16.0)
    with pytest.raises(MemoryError):
        v.transcribe(wav, voxtral_repo="models/voxtral-small-8bit", chunk_sec=60)
    # A huge machine still gets the shorter of the two caps, and the warning has
    # to name the one that actually bound. Saying "more context than the model
    # has" about a window the model has plenty of context for is worse than
    # saying nothing.
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 128.0)

    def pin(seconds):
        warnings.clear()
        with pytest.raises(_StopRun):
            v.transcribe(wav, voxtral_repo="models/voxtral-mini-8bit",
                         chunk_sec=seconds, need_timestamps=False,
                         log_cb=lambda lvl, msg: warnings.append((lvl, msg)))
        return [msg for lvl, msg in warnings
                if lvl == "warn" and "voxtral_chunk_sec" in msg]

    msgs = pin(2400)
    assert any(str(v.TRUSTED_CHUNK_SEC) in m and "measured" in m for m in msgs), msgs
    assert not any("context" in m for m in msgs), msgs

    # With the measured cap lifted past the context cap, the context wording is
    # the correct one again.
    monkeypatch.setattr(v, "TRUSTED_CHUNK_SEC", v.MAX_CHUNK_SEC + 600)
    msgs = pin(2400)
    assert any("context" in m and str(v.MAX_CHUNK_SEC) in m for m in msgs), msgs


def test_low_reserve_is_not_a_refusal(monkeypatch):
    """voxtral_ram_reserve_gb below MIN_HEADROOM_GB is the documented way to
    use a freed-up machine; it must clamp to the hard ceiling, not refuse a
    model that fits (regression: est_peak ~= total - reserve tripped the
    refusal for every reserve < 6)."""
    from noScribe import voxtral_engine as v

    monkeypatch.setattr(v, "_total_ram_gb", lambda: 32.0)
    # mini bf16 profile on 32 GB with reserve 5 used to raise MemoryError
    chunk = v._auto_chunk_sec("models/voxtral-mini", None, ram_reserve_gb=5)
    assert chunk > 0
    # and never beyond the hard ceiling: peak stays under total - MIN_HEADROOM
    m = v.MEM_MODEL["mini"]
    assert m["fixed"] + m["slope"] * chunk <= 32.0 - v.MIN_HEADROOM_GB + 0.01


def test_unquantised_source_repo_is_refused(monkeypatch, tmp_path):
    """The raw mistralai releases are conversion *sources*; the engine must
    refuse them instead of downloading tens of GB it then meters with the wrong
    memory profile."""
    import pytest
    from noScribe import voxtral_engine as v

    loaded = []
    monkeypatch.setattr(v, "_Voxtral", lambda repo: loaded.append(repo))
    for src in v.SOURCE_REPOS:
        with pytest.raises(ValueError, match="quantize_voxtral"):
            v.transcribe(_tiny_wav(tmp_path), voxtral_repo=src)
    assert loaded == []


def test_dense_encoder_builds_classify_by_bit_width():
    """The shipped builds keep the encoder in bf16 ("dense-encoder"); that adds
    under a GB, so they must classify by their bit width -- both as a local dir
    and as the published hub repo id downloaded on first use."""
    from noScribe import voxtral_engine as v

    assert v._model_kind("models/voxtral-mini-8bit") == "mini8"
    assert v._model_kind("MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder") == "mini8"
    assert v._model_kind("MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder") == "small"
    assert v._model_kind("MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder") == "small8"
    # older uniform names still map correctly
    assert v._model_kind("models/voxtral-small-6bit") == "small6"


def test_bare_api_default_model_exists():
    """transcribe() without voxtral_repo must fall back to a key that is
    actually in VOXTRAL_MODELS (the mini rename broke this silently once)."""
    from noScribe import voxtral_engine as v
    assert "voxtral-mini-8bit" in v.VOXTRAL_MODELS


def test_published_builds_are_offered(monkeypatch):
    """Both shipped builds download on first use, so has_local_build is true
    even with no local copy; an unknown name still needs a local build."""
    from noScribe import voxtral_engine as v

    monkeypatch.setattr(v, "_local_copy", lambda name: None)
    assert v.has_local_build("voxtral-mini-8bit")
    assert v.has_local_build("voxtral-small-4bit")
    assert not v.has_local_build("some-unbuilt-experiment")

    monkeypatch.setattr(v, "_local_copy", lambda name: f"models/{name}")
    assert v.has_local_build("some-unbuilt-experiment")


def test_split_sentences_keeps_unpunctuated_tail():
    """A pass whose text does not end in . ! ? must keep ALL its words, not just
    the last one. An earlier `\\S+$` fallback dropped everything between the last
    period and the final token (silent transcript loss on the .txt path)."""
    from noScribe.voxtral_engine import _split_sentences
    assert _split_sentences("das ist ein test ohne punkt") == [
        "das ist ein test ohne punkt"]
    assert _split_sentences("Hallo. Wie geht es dir") == [
        "Hallo.", "Wie geht es dir"]
    # normal punctuated text is unchanged
    assert _split_sentences("Ein Satz. Noch einer.") == ["Ein Satz.", "Noch einer."]


def test_model_kind_unknown_is_conservative():
    """An unrecognised build (no mini/small/size token) must fall back to the
    most memory-hungry profile, never the cheap `mini` one -- under-sizing only
    runs slower, over-sizing swaps forever."""
    from noScribe.voxtral_engine import _model_kind
    assert _model_kind("voxtral-mini-8bit") == "mini8"
    assert _model_kind("voxtral-small-8bit") == "small8"
    assert _model_kind("voxtral-small-4bit") == "small"
    assert _model_kind("totally-unknown-build") == "small8"
    # a 24B build that forgot the "small" token is still metered as big
    assert _model_kind("my-voxtral-24b-8bit") == "small8"


# --------------------------------------------------------------------------- #
# Keeping the clean prefix of a looping pass
# --------------------------------------------------------------------------- #

CLEAN_PREFIX = (
    "Wir haben gestern lange über die eigentlichen Ziele gesprochen. "
    "Danach kam ziemlich unvermittelt die Frage nach den Werten auf. "
    "Ich fand diese Diskussion ausgesprochen aufschlussreich und ehrlich. "
    "Am Ende blieb trotzdem eine gewisse Unsicherheit im Raum stehen. "
    "Deshalb schauen wir uns die einzelnen Schritte heute genauer an. "
    "Meine Empfehlung wäre, zunächst mit einer kleinen Übung zu beginnen. "
    "Wer damit Schwierigkeiten hat, meldet sich einfach kurz bei mir. "
    "Später sprechen wir dann über die Erfahrungen aus der Praxis. "
    "Vielleicht ergibt sich daraus schon eine erste gemeinsame Richtung. "
    "Für heute soll uns dieser Einstieg aber erst einmal genügen."
)


class _LoopsLateVoxtral:
    """Transcribes a long window well for a while and only then falls into a
    loop -- the shape a real pass has. Shorter windows come out clean."""

    def __init__(self):
        self.calls = []
        self._i = 0

    def transcribe_array(self, audio, language, max_new_tokens=0, repetition_penalty=1.0,
                         token_cb=None, temperature=0.0, seed=None, info=None):
        from noScribe.voxtral_engine import SAMPLE_RATE
        dur = len(audio) / SAMPLE_RATE
        self.calls.append(round(dur))
        if dur > 150:
            return CLEAN_PREFIX + " " + "dass das, " * 40
        text = VARIED_TEXT[self._i % len(VARIED_TEXT)]
        self._i += 1
        return text


def _align_one_word_per_second(words, window):
    """Stand-in for forced alignment: one word per second, confidently."""
    return [{"word": w, "start": float(i), "end": float(i + 1), "prob": 0.9}
            for i, w in enumerate(words)]


def _align_by_spreading(words, window):
    """What _Aligner returns when real alignment is impossible: words spread
    evenly over the window, marked with prob 0.0."""
    return [dict(s, prob=0.0) for s in _align_one_word_per_second(words, window)]


def test_degenerate_span_finds_where_the_loop_starts():
    from noScribe.voxtral_engine import _degenerate_span

    words = ("Ein ganz normaler Satz mit Inhalt. " + "dass das, " * 40).split()
    start, end = _degenerate_span(words)
    assert start == 6                      # right after the clean sentence
    assert end == len(words)
    # a cycle that repeats too few times is not a loop
    assert _degenerate_span("dass das, dass das, dass das,".split()) is None


def test_prefix_ends_at_the_last_completed_sentence():
    from noScribe.voxtral_engine import _prefix_end_index

    words = "Erster Satz. Zweiter Satz. Und dann kippt es weg weg weg".split()
    # the loop starts at "weg" (index 8); the unfinished sentence before it goes too
    assert _prefix_end_index(words, 8) == 4
    # nothing to keep when no sentence ends before the loop
    assert _prefix_end_index("und dann weg weg weg".split(), 2) == 0


def test_clean_prefix_is_kept_and_only_the_rest_is_redone():
    """The point of the rung: no second decode of the whole window, and the
    part the model got right is preserved verbatim rather than re-rolled."""
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _LoopsLateVoxtral()
    out = _transcribe_guarded(vox, _fake_audio(200), "de", None, "Pass 1/1",
                              align_cb=_align_one_word_per_second)

    assert not _looks_degenerate(out)
    assert "dass das," not in out
    assert "Deshalb schauen wir uns die einzelnen Schritte heute genauer an." in out
    # exactly two decodes: the original pass, then only what came after the loop
    assert len(vox.calls) == 2, vox.calls
    assert vox.calls[0] == 200
    assert 100 <= vox.calls[1] <= 106, vox.calls   # ~96s kept, remainder redone


def test_no_salvage_when_alignment_only_spread_the_words():
    """Spread positions are far too rough to cut audio on -- a wrong cut
    duplicates or drops speech, so the ladder must fall back instead."""
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    logged = []
    vox = _LoopsLateVoxtral()
    out = _transcribe_guarded(vox, _fake_audio(200), "de",
                              lambda level, msg: logged.append(msg), "Pass 1/1",
                              align_cb=_align_by_spreading)
    assert not _looks_degenerate(out)
    assert vox.calls[:2] == [200, 200]     # no salvage; the full retry happened
    # ...and the log states the reason, so a refused salvage can be told from
    # a rung that never ran at all
    assert any("cannot keep the clean part" in m and "even guess" in m
               for m in logged), logged


def test_declining_because_the_clean_part_is_short_says_so():
    from noScribe.voxtral_engine import _transcribe_guarded, SALVAGE_MIN_PREFIX_SEC

    logged = []
    _transcribe_guarded(_FakeVoxtral(), _fake_audio(200), "de",
                        lambda level, msg: logged.append(msg), "Pass 1/1",
                        align_cb=_align_one_word_per_second)
    assert any(f"below the {SALVAGE_MIN_PREFIX_SEC}s minimum" in m for m in logged), logged

def test_no_salvage_when_the_clean_part_is_too_short():
    """Below the minimum a full retry costs about the same and keeps the whole
    window's context, which a resumed pass loses."""
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _FakeVoxtral()                   # loops after five words
    out = _transcribe_guarded(vox, _fake_audio(200), "de", None, "Pass 1/1",
                              align_cb=_align_one_word_per_second)
    assert not _looks_degenerate(out)
    assert [d for d, _ in vox.calls] == [200, 200, 100, 100]   # unchanged ladder


ENGLISH_PREFIX = (
    "We spent a long time yesterday talking about what the actual goals are. "
    "After that there was a rather sudden question about the values. "
    "I thought that discussion was remarkably insightful and honest. "
    "In the end a certain uncertainty was still left in the room. "
    "That is why we are going to look at the individual steps more closely. "
    "My recommendation would be to start with a small exercise first. "
    "Anyone who has trouble with it should just get in touch with me. "
    "Later on we will talk about the experiences from actual practice. "
    "Perhaps a first shared direction will already come out of that. "
    "For today this introduction should be enough for us though."
)


class _TranslatesThenLoops:
    """The measured case from run 7: the greedy pass translates the window
    into English and then loops; every shorter pass comes back German. That
    very pass is the one the prefix rung keeps."""

    def __init__(self):
        self.calls = []

    def transcribe_array(self, audio, language, max_new_tokens=0, repetition_penalty=1.0,
                         token_cb=None, temperature=0.0, seed=None, info=None):
        from noScribe.voxtral_engine import SAMPLE_RATE
        dur = len(audio) / SAMPLE_RATE
        self.calls.append((round(dur), temperature))
        if dur > 150 and temperature == 0.0:
            return ENGLISH_PREFIX + " " + "dass das, " * 40
        return CLEAN_PREFIX


def test_a_translated_prefix_is_not_stitched_onto_a_german_remainder():
    """Voxtral occasionally translates instead of transcribing. All other
    rungs throw such a pass away and decode afresh -- which repairs the
    translation as a side effect. The prefix rung is the only one that KEEPS
    it, so it has to look: measured (run 7), the first 902 s of a chunk stayed
    English while the freshly transcribed remainder was German, and the
    transcript switched language in the middle of the chunk."""
    from noScribe.voxtral_engine import (_transcribe_guarded, _detect_language,
                                         _looks_degenerate)

    # Test setup: the two halves really have to be detected as different,
    # otherwise the test checks nothing.
    assert _detect_language(ENGLISH_PREFIX)[0] == "en"
    assert _detect_language(CLEAN_PREFIX)[0] == "de"

    logged = []
    vox = _TranslatesThenLoops()
    out = _transcribe_guarded(vox, _fake_audio(200), "de",
                              lambda level, msg: logged.append(msg), "Pass 1/1",
                              align_cb=_align_one_word_per_second)

    assert not _looks_degenerate(out)
    assert _detect_language(out)[0] == "de", \
        "the English prefix was stitched into the result"
    assert "welcome" not in out.lower() and "yesterday" not in out.lower()
    # ...and the reason is in the log, otherwise it looks like an ordinary loop
    assert any("translated pass" in m for m in logged), logged


def test_a_salvage_whose_halves_agree_is_still_kept():
    """Counter-check: the guard compares prefix against remainder, not against
    the configured language. Speakers mixing German and English are the
    normal case in this material -- a guard that checked against the setting
    would refuse good salvages by the dozen."""
    from noScribe.voxtral_engine import _transcribe_guarded

    vox = _LoopsLateVoxtral()
    out = _transcribe_guarded(vox, _fake_audio(200), "en", None, "Pass 1/1",
                              align_cb=_align_one_word_per_second)
    assert out.startswith(CLEAN_PREFIX[:40])      # salvaged although 'en' is set
    assert vox.calls[:2] == [200, 104]            # prefix kept, remainder redone


# --------------------------------------------------------------------------- #
# A whole chunk comes back in the wrong language
# --------------------------------------------------------------------------- #
def test_one_chunk_never_establishes_the_files_language():
    """The most dangerous mistake would be to take the language of the FIRST
    chunk as the truth: if that is the translated one, every following chunk
    would be 'repaired' into the wrong language. Two have to agree."""
    from noScribe.voxtral_engine import _file_language

    assert _file_language({}) is None
    assert _file_language({"en": 1}) is None            # the outlier alone
    assert _file_language({"en": 1, "de": 1}) is None   # a tie decides nothing
    assert _file_language({"en": 1, "de": 2}) == "de"
    assert _file_language({"de": 5, "en": 1}) == "de"


class _TranslatesWithoutLooping:
    """The measured case from runs 1-4: the greedy pass comes back English,
    without any loop at all. Only an attempt with temperature yields
    German."""

    def __init__(self):
        self.calls = []

    def transcribe_array(self, audio, language=None, max_new_tokens=0,
                         repetition_penalty=1.0, token_cb=None, temperature=0.0,
                         seed=None, info=None):
        self.calls.append(temperature)
        return ENGLISH_PREFIX if temperature == 0.0 else CLEAN_PREFIX


def test_a_translated_pass_makes_the_ladder_run_even_without_a_loop():
    """A translation is just as unusable as a loop and gets the same repairs
    -- the case needs no second ladder alongside. Measured: writing `lang:de`
    into the prompt brought back 0 of 2 flipped windows, one temperature
    attempt brought back both."""
    from noScribe.voxtral_engine import (_transcribe_guarded, _detect_language,
                                         _looks_degenerate)

    assert _detect_language(ENGLISH_PREFIX)[0] == "en"   # test setup
    logged = []
    vox = _TranslatesWithoutLooping()
    out = _transcribe_guarded(vox, _fake_audio(200), "de",
                              lambda lvl, msg: logged.append(msg), "Chunk 5/15",
                              want_lang="de")

    assert not _looks_degenerate(out)
    assert _detect_language(out)[0] == "de", "the translation was shipped"
    assert vox.calls[:2] == [0.0, 0.2], vox.calls
    assert any("translated pass" in m for m in logged), logged


def test_without_an_expected_language_nothing_is_called_translated():
    """Counter-check: as long as the file has not said what it is, no pass may
    be discarded because of its language -- otherwise the first chunk decides
    for all that follow."""
    from noScribe.voxtral_engine import _transcribe_guarded

    vox = _TranslatesWithoutLooping()
    out = _transcribe_guarded(vox, _fake_audio(200), "de", None, "Chunk 1/15")
    assert out == ENGLISH_PREFIX and vox.calls == [0.0]


def test_a_seam_between_two_languages_is_refused():
    """Both rungs that repair a window sew together two separately produced
    halves. If one of them flips into English, the transcript switches
    language in the middle of the chunk -- measured in run 7, where the kept
    prefix was 902 s of English and the freshly transcribed remainder German."""
    from noScribe.voxtral_engine import _halves_disagree

    logged = []
    log = lambda lvl, msg: logged.append(msg)
    assert _halves_disagree(ENGLISH_PREFIX, CLEAN_PREFIX, log, "Chunk 1/15",
                            "the kept part", "the remainder")
    assert any("translated pass" in m for m in logged), logged
    # Same language on both sides, and too short to judge: both stay silent.
    assert not _halves_disagree(CLEAN_PREFIX, CLEAN_PREFIX, log, "x", "a", "b")
    assert not _halves_disagree(CLEAN_PREFIX, "Ja, genau.", log, "x", "a", "b")


def test_transcribe_asks_the_ladder_for_a_language():
    """Wiring: the case from runs 1-4 had NO loop AT ALL. The expectation
    therefore has to go into the ladder, not be checked afterwards."""
    import inspect
    from noScribe import voxtral_engine as v

    src = inspect.getsource(v.transcribe)
    assert "want_lang=want" in src, "the ladder is not told the expected language"
    assert "_file_language(" in src, "the file language is not determined"
    # ...and whatever went out before the language became known is named.
    assert "already written out" in src


def test_remembered_model_survives_the_decorated_picker():
    """The model choice is stored as its plain name, not as the display text.

    The picker shows "name   ·   N GB RAM"; at start-up the remembered value is
    looked up in whisper_models, which holds plain names only. A merge had lost
    the model_key() call, which would have silently forgotten the choice at
    the next start."""
    import inspect
    import noScribe.main as m

    src = inspect.getsource(m.App.save_ui_state)
    assert "self.model_key(" in src, "save_ui_state stores the display text"
