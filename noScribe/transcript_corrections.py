"""
User-editable word corrections for transcripts.

Speech models (Voxtral in particular, which has no hotword support) reliably
mis-hear proper names — brands, products, programme names. This applies a
simple, predictable find/replace list the user maintains, e.g. turning
"Flor-Öle" into "VitaFlor" or "Sonvida" into "Sonvita".

The list lives in the noScribe config directory as `voxtral_corrections.yml`.
Format (case-insensitive, whole-word matches):

    - to: VitaFlor
      from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
"""

import functools
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

CORRECTIONS_FILENAME = "voxtral_corrections.yml"

DEFAULT_CORRECTIONS = """\
# noScribe - word corrections for the Voxtral engine
#
# Every "from" variant is replaced by the "to" value (case-insensitive,
# whole words/phrases only). Useful for brand, product and programme names
# the model keeps mis-hearing. One entry may list several spellings.
#
# Keep the variants specific. A short everyday phrase matches ordinary
# sentences too: "balance all" once ate the "Balance all dieser Faktoren".
#
# The file starts empty on purpose - corrections are personal to your
# material. Uncomment and adapt the examples to get started:
#
# - to: VitaFlor
#   from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
# - to: Sonvita
#   from: [sonvida, sonvieda, sonwita, "son vita"]
"""


def ensure_default_file(config_dir):
    """Create the corrections file with a documented default if it is missing.
    Returns the path to the file."""
    path = os.path.join(config_dir, CORRECTIONS_FILENAME)
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CORRECTIONS)
        except Exception as e:
            logger.warning("Could not create corrections file %s: %s", path, e)
    return path


def load_corrections(path):
    """Load the corrections file into a list of (compiled_regex, replacement)."""
    if not path or not os.path.exists(path):
        return []
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
    except Exception as e:
        logger.warning("Could not read corrections file %s: %s", path, e)
        return []
    if not isinstance(data, list):
        logger.warning("Corrections file %s must contain a list of entries.", path)
        return []

    rules = []
    for entry in data:
        # Tell the user about a mistyped entry instead of silently skipping it —
        # this file is hand-maintained and typos would otherwise look like the
        # feature not working.
        if not isinstance(entry, dict):
            logger.warning("Ignoring invalid corrections entry (not a mapping): %r", entry)
            continue
        to = entry.get("to")
        frm = entry.get("from") or []
        if isinstance(frm, str):
            frm = [frm]
        frm = [f for f in frm if f]
        if not to or not frm:
            logger.warning("Ignoring incomplete corrections entry (needs 'to' and 'from'): %r", entry)
            continue
        # Longest patterns first so "vita flor oil" wins over "vita flor".
        pats = sorted((re.escape(str(f)) for f in frm), key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(?:" + "|".join(pats) + r")(?!\w)", re.IGNORECASE)
        rules.append((rx, str(to)))
    return rules


# Spellings that sound the same (see _spelling_key): a speech model that hears
# a name it does not know picks one of them at random. Only these count. A
# looser rule -- any vowel for any other, m for n, g for k -- turned other
# people's names and fillers the dictionary does not know into a speaker's
# name ("Leni" -> "Lena", "Sena" -> "Sina", "Ähm" -> "Ann", Bavarian
# "Hamma" -> "Hanna"), so a mis-hearing that changes a sound ("Muna" for
# "Mona") is left to the user's correction list.
#
# Measured over 407k words of transcripts (Whisper and Voxtral output of
# CallHome German, English and Spanish and of AMI, plus private German
# recordings: a podcast, an interview, a video call), each checked against
# 70-150 common first names of its
# language and the names of its own speakers: one change, and a right one --
# a speaker's name the model had spelled with i for y. The dictionary kept
# the only other candidate ("Monica" beside "Monika") twice.
_SAME_SOUND = (
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"c(?=[aoulr]|$)"), "k"),       # Marcus / Markus
    (re.compile(r"dt$"), "t"),                  # Schmidt / Schmit
    (re.compile(r"(?<=.)y$"), "i"),             # Steffy / Steffi; inside a German
                                                # name y is often ü (Sybille)
    # an h after a vowel and before none: Mohna / Mona, Noah / Noa; not Johanna
    (re.compile(r"(?<=[aeiouäöü])h(?![aeiouäöü])"), ""),
)
# Doubled letters are deliberately not among them: a doubled vowel is another
# sound in Dutch, Finnish and English ("Joon" is not "John"), and a doubled
# consonant shortens the vowel before it in German and Italian, so "Ela" and
# "Ella", "Mila" and "Milla" are different names that sound different.


def _spelling_key(word):
    """The part of a spelling that can be heard: two words with the same key
    sound alike to anyone who does not know how either is written. Accents
    count ("Ole" is not "Öle"), and so does every vowel."""
    w = unicodedata.normalize("NFC", word).lower()
    for pattern, same in _SAME_SOUND:
        w = pattern.sub(same, w)
    return w


@functools.lru_cache(maxsize=None)
def _spell_checker():
    """macOS's own spell checker, or None.

    Voxtral runs only on Apple Silicon, and pyobjc is already a dependency
    there (environments/requirements_macOS_arm64.txt). It works in the spawned
    worker without a running app.
    """
    try:
        from AppKit import NSSpellChecker
        return NSSpellChecker.sharedSpellChecker()
    except Exception as e:
        logger.debug("No spell checker: %s", e)
        return None


@functools.lru_cache(maxsize=None)
def _dictionary_language(language):
    """The spell checker's name for a language code ("pt" -> "pt_BR"), or None
    when it has no dictionary for it. Asked about a language it lacks, the
    checker calls every word correct, so this must be settled up front."""
    checker = _spell_checker()
    code = str(language or "").strip().lower()[:2]
    if checker is None or len(code) != 2:
        return None
    try:
        available = [str(lang) for lang in checker.availableLanguages()]
    except Exception as e:     # a correction is never worth a failed job
        logger.debug("No dictionary list: %s", e)
        return None
    if code in available:
        return code
    return next((lang for lang in available if lang.startswith(code + "_")), None)


def _is_word(word, language):
    """Whether the dictionary of `language` (see _dictionary_language) knows
    `word`. Anything short of a clear "unknown" for the whole word counts as
    known: that leaves the name as the model wrote it, never a real word lost.

    Every way a lookup goes wrong ends up there. Under heavy load the spell
    server can stall ("NSSpellServer findMisspelledWordInString timed out"
    on stderr), and measured by pausing it, a stalled lookup reports no
    misspelling after ~1 s. A dictionary's first lookups while it loads, and
    every lookup in the 15 listed dictionaries that accept any word at all
    (uk, bg, el, he, hi, id, is, ga, ko, lt, nb, nn, pa, sl, te on macOS 27),
    answer the same way. Answers are therefore not cached across calls, so a
    stall costs at most one chunk's corrections; an ordinary lookup takes
    1-15 ms, and only the rare word that sounds like a name is asked.
    """
    try:
        miss, _count = _spell_checker().checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
            word, 0, language, False, 0, None)
    except Exception as e:
        logger.debug("Spell check of %r failed: %s", word, e)
        return True
    return not (miss.location == 0 and miss.length == len(word))


def apply_name_corrections(text, names, languages=None):
    """Set mis-heard spellings of the speakers' names to the one the user gave.

    A speech model hears a name it does not know and writes one of the ways
    it could be spelled ("Mohna" for "Mona", "Steffy" for "Steffi"). Where the
    user has told us the names (the speaker-names field), that can be fixed.

    A capitalised word is replaced only when it passes both guards:
      * it is spelled differently from exactly one of the names but sounds the
        same (`_spelling_key`),
      * and no dictionary of `languages` -- one code or several, e.g. the
        language the text reads as and the one the file is in -- knows it.
    The dictionary keeps real words that happen to be spelled like a name
    (every German noun is capitalised), and it knows common names, so
    "Marcus" stays "Marcus" next to a speaker called Markus: a real name may
    be someone else. Asking every language the text may be in keeps a word of
    one language out of the other's gaps, such as an English quote in a German
    passage. The measurement is above _SAME_SOUND.

    Without a dictionary (not macOS, Auto before the language is known, or a
    language the system has none for) nothing is changed -- and nothing is
    either with one of the dictionaries that accept every word (see _is_word).
    """
    if isinstance(languages, str) or languages is None:
        languages = [languages]
    dictionaries = list(dict.fromkeys(
        d for d in (_dictionary_language(code) for code in languages) if d))
    if not dictionaries:
        return text
    # "Mona Muster" or "Anna-Lena" as a speaker name: each capitalised part is
    # a name of its own ("del" in "Ana del Río" is not).
    parts = {p for n in (names or [])
             for p in re.findall(r"[^\W\d_]{3,}", unicodedata.normalize("NFC", str(n)))
             if p[0].isupper()}
    keys = {}
    for p in parts:
        keys.setdefault(_spelling_key(p), set()).add(p)
    lowered = {p.lower() for p in parts}
    verdicts = {}   # one lookup per word and chunk (see _is_word)

    def repl(m):
        word = m.group(0)
        if not word[0].isupper() or word.lower() in lowered:
            return word
        key = _spelling_key(word)
        alike = keys.get(key, ())
        if len(alike) != 1:
            return word
        if word not in verdicts:
            verdicts[word] = any(_is_word(word, d) for d in dictionaries)
        return word if verdicts[word] else next(iter(alike))

    # Not the stem of a contraction or possessive ("Weren't", "Mohna's").
    return re.sub(r"(?<!\w)[^\W\d_]{3,}(?![\w'\u2019])", repl, text)


def apply_corrections(text, rules):
    """Apply the compiled correction rules to `text`."""
    for rx, to in rules:
        # A callable replacement treats `to` as literal text — a backslash or
        # "\1" in a user-supplied value must never be a regex template.
        text = rx.sub(lambda m, to=to: to, text)
    return text
