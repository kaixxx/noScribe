"""When a diarization label deserves a second look.

The thresholds are measured against ground truth, not guessed: VoxConverse
v0.3, 216 dev plus 232 test recordings with 1 to 21 speakers and an RTTM
reference. At label level the precision is about two thirds (dev 83 %, test
69 %), the recall about one in eight. Both figures are recorded here because
they bound what the report may claim -- and because a tighter threshold
demonstrably does NOT help: 0.01 scored 100 % on dev and 67 % on test.

The counter-check that justifies the second condition: on a 25-minute question
round with 14 real speakers, 13 hold less than 5 % of the speech time -- but
every one of them speaks in sentences, and across VoxConverse dev only 2 of 860
real speakers ever stay under a 2 s turn.
"""
import yaml
from pathlib import Path

from noScribe.main import (
    GHOST_SPEAKER_MAX_SHARE,
    GHOST_SPEAKER_MAX_TURN_MS,
    find_ghost_speakers,
)


def _seg(start, end, label):
    return {'start': int(start * 1000), 'end': int(end * 1000), 'label': label}


def _spread(label, n, length, step=20.0, offset=0.0):
    """n short turns spread over the file -- the shape of a ghost."""
    return [_seg(offset + i * step, offset + i * step + length, label)
            for i in range(n)]


def _floor(label, n, length, step=60.0, offset=5.0):
    """n real turns holding the floor."""
    return [_seg(offset + i * step, offset + i * step + length, label)
            for i in range(n)]


def test_the_measured_ghost_is_reported():
    """The real case: two speakers at ~50 % each with turns up to 29 s, plus a
    label with 1.7 % and no turn over 1.52 s -- both under the thresholds."""
    diarization = (_floor('SPEAKER_00', 20, 28.0)
                   + _floor('SPEAKER_01', 20, 26.0, offset=35.0)
                   + _spread('SPEAKER_02', 56, 0.35))
    ghosts = find_ghost_speakers(diarization)
    assert [g[0] for g in ghosts] == ['SPEAKER_02'], ghosts
    label, share, longest = ghosts[0]
    assert share < 0.05 and longest <= 1520


def test_fourteen_real_speakers_are_left_alone():
    """The false-positive control, and the reason for the second condition:
    13 of the 14 measured speakers hold less than 5 % of the speech time. Their
    longest turn was between 5.86 s and 42.8 s, far above the threshold."""
    longest_turns = [5.855, 6.109, 7.830, 8.522, 10.125, 10.851, 11.104,
                     11.813, 15.086, 16.301, 22.950, 24.401, 28.620, 42.778]
    diarization = []
    for i, longest in enumerate(longest_turns):
        # one long turn plus small change -- the share stays small
        diarization.append(_seg(i * 100, i * 100 + longest, f'SPEAKER_{i:02d}'))
        diarization += _spread(f'SPEAKER_{i:02d}', 3, 0.4, offset=i * 100 + 50)
    diarization += _floor('SPEAKER_99', 12, 42.0, offset=2000.0)   # the trainer
    assert find_ghost_speakers(diarization) == []


def test_two_speakers_are_never_judged():
    """With two labels, "one of them is spurious" is not a conclusion this can
    draw -- the remaining one would have to be everybody. Not even when one of
    the two is tiny."""
    diarization = _floor('SPEAKER_00', 20, 28.0) + _spread('SPEAKER_01', 40, 0.3)
    assert find_ghost_speakers(diarization) == []


def test_not_every_label_can_be_a_ghost():
    """A file with no speaker at all is never the useful reading, however
    lopsided the distribution."""
    diarization = (_spread('SPEAKER_00', 30, 0.4)
                   + _spread('SPEAKER_01', 30, 0.4, offset=1.0)
                   + _spread('SPEAKER_02', 30, 0.4, offset=2.0))
    assert find_ghost_speakers(diarization) == []


def test_a_brief_but_real_third_speaker_is_left_alone():
    """The edge case the turn-length condition protects: someone answers one
    question and is otherwise silent. Little speech time, but a whole sentence."""
    diarization = (_floor('SPEAKER_00', 20, 28.0)
                   + _floor('SPEAKER_01', 20, 26.0, offset=35.0)
                   + [_seg(500.0, 504.5, 'SPEAKER_02')])
    assert find_ghost_speakers(diarization) == []


def test_empty_and_degenerate_input():
    assert find_ghost_speakers(None) == []
    assert find_ghost_speakers([]) == []
    # Every segment of zero length: no speech time, so nothing to judge.
    assert find_ghost_speakers([_seg(1.0, 1.0, f'SPEAKER_0{i}') for i in range(3)]) == []


def test_thresholds_are_the_documented_ones():
    """The values carry the report's claim; a quiet tweak turns a measured
    threshold into a guessed one. 0.02 is the only value that beat the original
    0.05 on dev AND test."""
    assert GHOST_SPEAKER_MAX_SHARE == 0.02
    assert GHOST_SPEAKER_MAX_TURN_MS == 2000


def test_a_quiet_real_speaker_can_trip_this_and_that_is_known():
    """The method's limit, recorded on purpose. On VoxConverse test about a third
    of the reports were a real speaker who simply said almost nothing -- measured
    e.g. 0.32 % of the speech time with 1266 ms as the longest turn, more extreme
    than the ghost that started all this (1.7 % / 1519 ms). On these two axes
    the two cannot be told apart; that is why the function reports and does not
    decide."""
    diarization = (_floor('SPEAKER_00', 20, 30.0)
                   + _floor('SPEAKER_01', 20, 28.0, offset=35.0)
                   + [_seg(400.0, 401.27, 'SPEAKER_02'),
                      _seg(700.0, 701.1, 'SPEAKER_02')])
    ghosts = find_ghost_speakers(diarization)
    assert [g[0] for g in ghosts] == ['SPEAKER_02'], ghosts


def test_many_short_turns_are_not_required():
    """The obvious third axis was measured and rejected: the ghost that started
    this had 56 short turns, but on VoxConverse surplus labels typically have
    few -- a minimum turn count dropped the hits from 5 to 1. So a label with
    only two short turns must still be reported."""
    diarization = (_floor('SPEAKER_00', 20, 30.0)
                   + _floor('SPEAKER_01', 20, 28.0, offset=35.0)
                   + [_seg(400.0, 400.9, 'SPEAKER_02'),
                      _seg(900.0, 901.4, 'SPEAKER_02')])
    assert [g[0] for g in find_ghost_speakers(diarization)] == ['SPEAKER_02']


def test_warning_string_exists_in_every_locale_or_falls_back_to_english():
    """i18n falls back to "en", so the key has to be there -- otherwise the log
    shows the key name instead of a sentence."""
    trans = Path(__file__).resolve().parent.parent / 'trans'
    en = yaml.safe_load((trans / 'noScribe.en.yml').read_text(encoding='utf-8'))
    assert 'warn_ghost_speaker' in en['en']
    for placeholder in ('%{speaker}', '%{share}', '%{longest}'):
        assert placeholder in en['en']['warn_ghost_speaker']
