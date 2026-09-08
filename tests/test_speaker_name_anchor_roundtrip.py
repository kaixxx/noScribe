"""The user's speaker names end up verbatim inside the audio-sync anchors
(``ts_{start}_{end}_{speaker}``), which several readers parse with different
assumptions -- ``noScribe.utils.html_to_webvtt`` here, noScribeEdit's VTT
export in the other application, plus the HTML attribute the anchor lives in.

``parse_speaker_names`` is the only guard for all of them, and that contract
lived in its docstring alone. These tests pin it against the real converter,
so a change to the sanitizer or to the anchor format fails here rather than in
someone's transcript.

Deterministic rather than randomised: the alphabet below is every character
with a role in one of the readers, which is small enough to cover exhaustively
while keeping CI reproducible.
"""
import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

from noScribe import utils
from noScribe.main import parse_speaker_names

# Characters with meaning to one of the anchor readers, plus the whitespace and
# quoting that surrounds them in the HTML.
HOSTILE = ['_', '<', '>', '"', '&', ':', "'", '\\', '/', '//', '  ', '\t', '\n',
           '|', '%', '=', '#', '?']
# Names people plausibly type, including ones that collide with the above.
PLAUSIBLE = ['Mona', 'Anna-Lena', 'Dr. Muster', 'Mona M.', 'José', 'Æsa', '李雷']

TEXT = 'Guten Morgen, alle zusammen.'


def _names_under_test():
    for h in HOSTILE:
        yield from (h, f'Mona{h}Lena', f'{h}Mona', f'Mona{h}')
    for p in PLAUSIBLE:
        yield from (p, f'{p}_{p}')


def _cue_for(name):
    """Run a one-segment transcript through the real VTT converter, built the
    way on_segment writes it with timestamps on -- the form where the speaker
    string is used twice: once in the anchor, once as the "Name:" label that
    the converter has to recognise and drop."""
    seg = (f'<a name="ts_1000_2000_{name}" >{name}: '
           f'<span style="color: #78909C" >[00:00:01]</span>{TEXT}</a>')
    vtt = utils.html_to_webvtt(f'<html><body><p>T</p><p>I</p><p>{seg}</p></body></html>')
    return vtt.splitlines()[-2]


@pytest.mark.parametrize('raw', list(_names_under_test()))
def test_sanitized_name_survives_the_anchor_roundtrip(raw):
    names = parse_speaker_names(raw)
    if not names:  # input sanitized away entirely -- nothing reaches an anchor
        return
    name = names[0]

    # None of the characters that give the anchor its structure may survive.
    # The underscore is in this set for noScribeEdit's sake: it splits the
    # anchor with a limit of 4 (noScribeEdit.py, html_to_vtt), so a name
    # containing one would be truncated there even though the converter
    # exercised below handles it.
    assert not set(name) & set('_<>"&:'), f'{name!r} still breaks an anchor'

    cue = _cue_for(name)
    # The converter recovers exactly the name that went in ...
    assert cue.startswith(f'<v {name}>'), f'{name!r} did not survive the anchor'
    # ... and recognises the label well enough to drop it from the cue text,
    # which is a second, independent use of the same string.
    assert cue == f'<v {name}>{TEXT}'


@pytest.mark.parametrize('raw, damage', [
    ('Dr: Muster', 'a colon defeats the label stripping'),
    ('Mona" x="y', 'a quote closes the attribute early'),
    ('<Boss>', 'angle brackets corrupt the voice span'),
])
def test_the_guard_is_what_makes_the_roundtrip_work(raw, damage):
    """Feed the raw name straight through, as would happen without the
    sanitizer, and show it really does corrupt the cue -- otherwise the
    assertions above would be vacuously true."""
    assert _cue_for(raw) != f'<v {raw}>{TEXT}', f'expected damage: {damage}'


def test_overlap_marker_reads_as_the_mapped_name():
    """An overlapping turn carries '//' in front of the name; that prefix is
    the marker, and it belongs to the speaker field."""
    assert _cue_for('//Mona').startswith('<v //Mona>')
