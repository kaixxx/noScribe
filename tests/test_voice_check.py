"""noScribe.voice_check: units, the two margins, and the passages a rewrite is made of.

The thresholds themselves are justified in the module's docstring (word-level
speaker error against ground truth). These tests pin the *mechanics* the
measurement relied on, so a refactor cannot quietly change what was measured:
where a unit ends, which margin applies when, and that writing the passages
again reproduces the transcript wherever nothing moved.
"""
import math

import pytest

from noScribe import voice_check as vc


def words(*items):
    """('text', start, end) -> faster-whisper style words (leading space)."""
    return [{'word': ' ' + text, 'start': start, 'end': end} for text, start, end in items]


def said(text, start, end):
    """A segment that is one word long."""
    return {'start': start, 'end': end, 'text': ' ' + text, 'words': words((text, start, end))}


A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
CENTROIDS = {'S00': A, 'S01': B}


def voice(towards_b):
    """An embedding whose cosine margin for S01 over S00 is `towards_b`
    (sin x - cos x = sqrt(2) sin(x - pi/4))."""
    x = math.pi / 4 + math.asin(towards_b / math.sqrt(2))
    return [math.cos(x), math.sin(x), 0.0]


def relabel(segments, turns, centroids, embed):
    """The check as measured: the voice as its evidence."""
    return vc.relabel(segments, turns, vc.Voice(centroids, embed))


def scores(embedding, centroids=CENTROIDS):
    return vc._scores(embedding, centroids)


def speakers(passages):
    return [speaker for _, speaker in passages]


def test_a_unit_ends_at_a_sentence_end_and_at_a_long_pause():
    ws = words(('Is', 0.0, 0.2), ('that', 0.2, 0.4), ('so?', 0.4, 0.7),
               ('Yes.', 0.8, 1.0), ('and', 1.1, 1.3), ('then', 1.9, 2.1))
    units = vc.split_units(ws)
    assert [[w['word'].strip() for w in u] for u in units] == [['Is', 'that', 'so?'], ['Yes.'], ['and'], ['then']]


def test_a_short_pause_does_not_end_a_unit():
    ws = words(('and', 0.0, 0.2), ('then', 0.2 + vc.UNIT_PAUSE_S - 0.01, 1.0))
    assert len(vc.split_units(ws)) == 1


def test_an_abbreviation_only_offers_a_cut_the_voice_still_has_to_take_it():
    """"Dr." ends a unit, which is harmless: without a voice that says otherwise
    both units keep the segment's speaker and are written as one passage."""
    ws = words(('Frau', 0.0, 0.3), ('Dr.', 0.3, 0.6), ('Muster', 0.6, 1.0), ('kommt.', 1.0, 1.4))
    seg = {'start': 0.0, 'end': 1.4, 'text': ' Frau Dr. Muster kommt.', 'words': ws}
    passages, moves = relabel([(seg, 'S00', False)], [], CENTROIDS, lambda spans: [voice(-0.9)] * len(spans))
    assert moves == [] and passages == [(seg, 'S00')]


def test_words_without_stamps_leave_the_segment_alone():
    """No stamps, no span to take a voice from -- and above all no span with a
    None in it, which would cost the worker call for the whole recording."""
    ws = [{'word': ' Yes.', 'start': None, 'end': None}, {'word': ' No.', 'start': 1.0, 'end': 1.2}]
    seg = {'start': 0.0, 'end': 1.2, 'text': ' Yes. No.', 'words': ws}
    assert vc.split_units(ws) == []

    def embed(spans):
        raise AssertionError(f'nothing to embed, got {spans}')

    assert relabel([(seg, 'S00', False)], [], CENTROIDS, embed) == ([(seg, 'S00')], [])


def test_the_diarization_agreeing_needs_the_voice_to_lean_its_way_only():
    turns = {'S01': 400}
    assert vc.decide('S00', turns, scores(voice(vc.MARGIN_AGREE + 0.05))) == 'S01'
    assert vc.decide('S00', turns, scores(voice(vc.MARGIN_AGREE - 0.05))) == 'S00'


def test_the_voice_alone_needs_the_large_margin():
    turns = {'S00': 400}
    assert vc.decide('S00', turns, scores(voice(vc.MARGIN_OVERRULE + 0.05))) == 'S01'
    assert vc.decide('S00', turns, scores(voice(vc.MARGIN_OVERRULE - 0.05))) == 'S00'


def test_a_whole_segment_is_never_moved_on_the_small_margin():
    """Its diarization majority is what gave it its speaker in the first place --
    a different label inside it would be the overlap rule's doing, not news."""
    seg = said('Exactly.', 0.0, 0.6)
    turns = [{'start': 0, 'end': 600, 'label': 'S01'}]
    between = (vc.MARGIN_AGREE + vc.MARGIN_OVERRULE) / 2
    assert relabel([(seg, 'S00', False)], turns, CENTROIDS, lambda spans: [voice(between)]) == ([(seg, 'S00')], [])
    # ... while the same voice does move a unit that is only part of its segment
    ws = words(('Exactly.', 0.0, 0.6), ('Right.', 0.7, 1.2))
    two = {'start': 0.0, 'end': 1.2, 'text': ' Exactly. Right.', 'words': ws}
    turns = [{'start': 0, 'end': 600, 'label': 'S01'}, {'start': 700, 'end': 1200, 'label': 'S00'}]
    passages, moves = relabel([(two, 'S00', False)], turns, CENTROIDS, lambda spans: [voice(between), voice(-0.9)])
    assert speakers(passages) == ['S01', 'S00'] and [m[1:] for m in moves] == [('S00', 'S01')]


@pytest.mark.parametrize('embedding, centroids', [
    (None, CENTROIDS),                    # no audio to embed
    (voice(0.9), {'S01': B}),             # the current speaker has no centroid
    (voice(0.9), {'S00': A, 'S01': [0.0, 0.0, 0.0]}),  # a padded, empty centroid
])
def test_without_something_to_compare_nothing_moves(embedding, centroids):
    assert vc.decide('S00', {'S01': 400}, scores(embedding, centroids)) == 'S00'


def test_a_turn_final_answer_becomes_its_own_passage():
    ws = words(('Does', 0.0, 0.2), ('it', 0.2, 0.3), ('help?', 0.3, 0.8), ('Yes,', 0.9, 1.1), ('sure.', 1.1, 1.5))
    seg = {'start': 0.0, 'end': 1.5, 'text': ' Does it help? Yes, sure.', 'words': ws}
    turns = [{'start': 0, 'end': 850, 'label': 'S00'}, {'start': 880, 'end': 1500, 'label': 'S01'}]
    asked = []

    def embed(spans):
        asked.extend(spans)
        return [voice(-0.9), voice(0.5)]

    passages, moves = relabel([(seg, 'S00', False)], turns, CENTROIDS, embed)
    assert asked == [[0.0, 0.8], [0.9, 1.5]]
    assert [(p['text'], speaker) for p, speaker in passages] == [(' Does it help?', 'S00'), (' Yes, sure.', 'S01')]
    assert passages[1][0]['start'] == 0.9 and passages[1][0]['words'] == ws[3:]
    assert moves == [(passages[1][0], 'S00', 'S01')]


def test_a_segment_moved_as_a_whole_keeps_its_dict():
    seg = said('Exactly.', 0.0, 0.6)
    passages, moves = relabel([(seg, 'S00', False)], [], CENTROIDS, lambda spans: [voice(0.9)])
    assert passages == [(seg, 'S01')] and moves == [(seg, 'S00', 'S01')]


@pytest.mark.parametrize('tokens, text, pieces', [
    ([' Yes.', ' No', ' way.'], ' Yes. No way.', [' Yes.', ' No way.']),      # faster-whisper
    (['Yes.', 'No', 'way.'], ' Yes. No way.', [' Yes.', ' No way.']),         # bare tokens, text joins with spaces
    (['好。', '不', '行。'], '好。不行。', ['好。', '不行。']),                   # written without spaces
])
def test_a_passage_is_joined_the_way_its_segment_is(tokens, text, pieces):
    ws = [{'word': token, 'start': i * 0.5, 'end': i * 0.5 + 0.4} for i, token in enumerate(tokens)]
    seg = {'start': 0.0, 'end': 1.4, 'text': text, 'words': ws}
    passages, _ = relabel([(seg, 'S00', False)], [], CENTROIDS, lambda spans: [voice(-0.9), voice(0.9)])
    assert [p['text'] for p, _ in passages] == pieces


def test_one_speaker_means_no_embedding_call_at_all():
    seg = said('Exactly.', 0.0, 0.6)

    def embed(spans):
        raise AssertionError('the worker must not be started for nothing')

    assert relabel([(seg, 'S00', False)], [], {'S00': A}, embed) == ([(seg, 'S00')], [])


def test_an_untouched_segment_keeps_its_overlap_marker():
    """find_speaker writes overlapping speech as '//S01'. The voice is compared
    under the plain label, and the segment is handed back as it was written."""
    seg = said('Exactly.', 0.0, 0.6)
    asked = []
    passages, moves = relabel([(seg, '//S01', False)], [], CENTROIDS,
                                 lambda spans: asked.extend(spans) or [voice(0.9)])
    assert asked and moves == [] and passages == [(seg, '//S01')]


def test_an_interjection_that_moves_to_a_third_voice_stays_an_interjection():
    """S00 holds the floor, the diarization heard S01 talk into it, the voice says
    S02: still someone talking into S00's turn, so the marker stays."""
    centroids = dict(CENTROIDS, S02=[0.0, 0.0, 1.0])
    voices = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    passages, moves = relabel([(said('So we went on.', 0.0, 2.0), 'S00', False), (said('Really?', 2.1, 2.6), '//S01', False)],
                                 [], centroids, lambda spans: voices)
    assert speakers(passages) == ['S00', '//S02'] and [m[1:] for m in moves] == [('//S01', '//S02')]


def test_an_interjection_that_is_the_floor_holders_own_voice_just_continues():
    passages, moves = relabel([(said('So we went on.', 0.0, 2.0), 'S00', False), (said('And on.', 2.1, 2.6), '//S01', False)],
                                 [], CENTROIDS, lambda spans: [voice(-0.9), voice(-0.9)])
    assert speakers(passages) == ['S00', 'S00'] and len(moves) == 1


def test_nobody_talks_into_their_own_turn_after_a_move():
    """The floor holder moves to S01; the aside the diarization gave S01 did not
    move, and is now simply S01 going on."""
    passages, moves = relabel([(said('I disagree.', 0.0, 2.0), 'S00', False), (said('Completely.', 2.1, 2.9), '//S01', False)],
                                 [], CENTROIDS, lambda spans: [voice(0.9), voice(0.9)])
    assert speakers(passages) == ['S01', 'S01'] and len(moves) == 1


def test_a_segment_the_diarization_was_silent_on_follows_its_predecessor():
    """It was written under S01 only because its predecessor was. Handing it back
    with '' keeps it that way in the rewrite -- also when the predecessor moves,
    and the move is on record."""
    first, gap = said('And on again.', 0.0, 2.0), said('Mhm right', 2.1, 2.9)
    passages, moves = relabel([(first, 'S01', False), (gap, 'S01', True)], [], CENTROIDS,
                                 lambda spans: [voice(-0.9), voice(0.0)])
    assert speakers(passages) == ['S00', ''] and [m[1:] for m in moves] == [('S01', 'S00'), ('S01', 'S00')]


def test_an_inherited_segment_with_a_voice_of_its_own_keeps_it():
    """Held against the speaker it would inherit NOW: the predecessor went to S00,
    this one is clearly S01, so it says so instead of tagging along."""
    first, gap = said('And on again.', 0.0, 2.0), said('Mhm right', 2.1, 2.9)
    passages, moves = relabel([(first, 'S01', False), (gap, 'S01', True)], [], CENTROIDS,
                                 lambda spans: [voice(-0.9), voice(0.9)])
    assert speakers(passages) == ['S00', 'S01'] and len(moves) == 1


def test_only_the_start_of_an_inherited_segment_may_follow():
    """After a moved unit, '' would inherit the moved speaker: what comes back to
    the segment's own speaker has to say so."""
    ws = words(('One.', 0.0, 0.5), ('Two.', 0.6, 1.1), ('Three.', 1.2, 1.7))
    seg = {'start': 0.0, 'end': 1.7, 'text': ' One. Two. Three.', 'words': ws}
    passages, moves = relabel([(said('Go on.', -1.0, -0.2), 'S00', False), (seg, 'S00', True)], [], CENTROIDS,
                                 lambda spans: [voice(-0.9), voice(-0.9), voice(0.9), voice(-0.9)])
    assert speakers(passages) == ['S00', '', 'S01', 'S00'] and [m[2] for m in moves] == ['S01']


def test_writing_the_passages_again_changes_nothing_where_nothing_moved():
    """Marked, inherited and speakerless segments all come back as they were
    written -- the inherited ones as '' -- and nothing is on record as moved."""
    segs = [(said('Um.', 0.0, 0.4), '', False), (said('Hello.', 0.5, 1.0), '//S01', False),
            (said('Yes.', 1.1, 1.5), '//S01', True), (said('Well.', 1.6, 2.0), 'S00', False),
            (said('So.', 2.1, 2.5), 'S00', True), (said('Hm.', 2.6, 3.0), '//S01', False)]
    own = {'S00': voice(-0.9), 'S01': voice(0.9)}
    passages, moves = relabel(segs, [], CENTROIDS,
                                 lambda spans: [own[w.lstrip('/')] for _, w, _ in segs if w])
    assert moves == []
    assert passages == [(seg, '' if inherited else written) for seg, written, inherited in segs]


@pytest.mark.parametrize('value, expected', [('0', False), ('off', False), ('1', True), ('', True)])
def test_the_environment_switch(monkeypatch, value, expected):
    monkeypatch.setenv('NOSCRIBE_VOICE_CHECK', value)
    assert vc.enabled() is expected


@pytest.mark.parametrize('setting, expected', [
    (False, False), ('False', False), (0, False), ('0', False), ('no', False), (' Off ', False),
    (True, True), ('True', True), ('1', True), (None, True)])
def test_the_config_switch_takes_what_the_environment_takes(monkeypatch, setting, expected):
    """config.yml's `voice_check` used to be off only for a literal false, while
    the environment switch also took 0, no and off: `voice_check: 0` or `off`
    left the check running."""
    monkeypatch.delenv('NOSCRIBE_VOICE_CHECK', raising=False)
    assert vc.enabled(setting) is expected
    monkeypatch.setenv('NOSCRIBE_VOICE_CHECK', 'no')
    assert vc.enabled(setting) is False


def test_turns_inside_finds_what_a_scan_from_the_start_finds():
    """turns_inside skips the turns that end before the unit, by bisection;
    turns that nest inside a long one are what could make such a skip miss one.
    Held against the plain scan from the first turn it replaced."""
    import random

    def scan(diarization, start_ms, end_ms):
        totals = {}
        for turn in diarization:
            if turn['start'] > end_ms:
                break
            inside = min(turn['end'], end_ms) - max(turn['start'], start_ms)
            if inside > 0:
                totals[turn['label']] = totals.get(turn['label'], 0) + inside
        return totals

    for seed in range(300):
        rnd = random.Random(seed)
        turns = []
        for _ in range(rnd.randint(0, 40)):
            start = rnd.randint(0, 60000)
            turns.append({'start': start, 'end': start + rnd.choice((0, rnd.randint(1, 800), rnd.randint(1, 20000))),
                          'label': f'S0{rnd.randint(0, 3)}'})
        turns.sort(key=lambda turn: turn['start'])
        for _ in range(30):
            a = rnd.randint(-1000, 62000)
            b = a + rnd.choice((0, rnd.randint(1, 3000)))
            assert vc.turns_inside(turns, a, b) == scan(turns, a, b), (seed, a, b)


def test_a_recording_of_similar_voices_gets_smaller_margins():
    """Long passages favour their own speaker by only 0.3 here (two similar voices
    on one channel), where a margin measured on voices far apart asks too much:
    both margins shrink to 0.3 / MARGIN_SCALE_REF of their value."""
    long_ones = [(said(f'Long passage {i}.', 10.0 * i, 10.0 * i + 3.0), 'S00', False) for i in range(3)]
    aside = (said('Exactly.', 40.0, 40.6), 'S00', False)
    lean = vc.MARGIN_OVERRULE * 0.8
    voices = [voice(-0.3)] * 3 + [voice(lean)]
    passages, moves = relabel(long_ones + [aside], [], CENTROIDS, lambda spans: voices)
    assert vc.margin_scale([('S00', scores(voice(-0.3)), 3.0)] * 3) == pytest.approx(0.3 / vc.MARGIN_SCALE_REF)
    assert speakers(passages)[-1] == 'S01' and len(moves) == 1
    # ... while the same aside stays put in a recording whose voices are far apart
    voices = [voice(-0.8)] * 3 + [voice(lean)]
    assert relabel(long_ones + [aside], [], CENTROIDS, lambda spans: voices)[1] == []


@pytest.mark.parametrize('units, expected', [
    ([('S00', voice(-0.1), 3.0)], vc.MARGIN_SCALE_FLOOR),       # never below the floor
    ([('S00', voice(-0.9), 3.0)], 1.0),                          # never above the measured margins
    ([('S00', voice(-0.3), 1.0)], 1.0),                          # short passages say nothing about the scale
    ([('S00', voice(0.4), 3.0)], 1.0),                           # nor do passages that favour someone else
    ([], 1.0),
])
def test_the_margin_scale_has_bounds_and_needs_evidence(units, expected):
    assert vc.margin_scale([(label, scores(voice_), seconds) for label, voice_, seconds in units]) == pytest.approx(expected)


def test_a_voice_too_short_to_say_much_still_follows_the_diarization():
    """'Perfekt.', 0.3 s, wholly inside the other speaker's turn, glued to the
    'Ja.' before it by the segment builder. 0.3 s of audio has no voice to speak
    of (+0.03), and demanding a margin before it may follow the diarization kept
    a word with the wrong speaker that the diarization had right."""
    ws = words(('Ja.', 0.0, 0.5), ('Perfekt.', 1.0, 1.3))
    seg = {'start': 0.0, 'end': 1.3, 'text': ' Ja. Perfekt.', 'words': ws}
    turns = [{'start': 0, 'end': 600, 'label': 'S00'}, {'start': 900, 'end': 5000, 'label': 'S01'}]
    passages, moves = relabel([(seg, 'S00', False)], turns, CENTROIDS, lambda spans: [voice(-0.5), voice(0.03)])
    assert speakers(passages) == ['S00', 'S01'] and len(moves) == 1
    # ... but a voice that leans the other way keeps it where it is
    assert relabel([(seg, 'S00', False)], turns, CENTROIDS, lambda spans: [voice(-0.5), voice(-0.03)])[1] == []



def test_a_word_without_length_goes_with_its_neighbour():
    """faster-whisper emits words whose start equals their end. The crop for one
    is widened into its neighbours, so its "voice" is theirs, and moved on its own
    it would become a subtitle cue from t to t, which WebVTT forbids."""
    ws = words(('So.', 0.0, 1.0), ('Hm.', 1.6, 1.6), ('Right.', 2.2, 3.0))
    seg = {'start': 0.0, 'end': 3.0, 'text': ' So. Hm. Right.', 'words': ws}
    turns = [{'start': 0, 'end': 3000, 'label': 'S00'}]
    passages, moves = relabel([(seg, 'S00', False)], turns, CENTROIDS,
                                 lambda spans: [voice(-0.9), voice(0.9), voice(0.9)])
    assert [(p['text'], s) for p, s in passages] == [(' So. Hm.', 'S00'), (' Right.', 'S01')]
    # ... and at the head of a segment, with the unit after it
    ws = words(('Hm.', 0.0, 0.0), ('Right.', 0.6, 1.5), ('So.', 2.1, 3.0))
    seg = {'start': 0.0, 'end': 3.0, 'text': ' Hm. Right. So.', 'words': ws}
    passages, _ = relabel([(seg, 'S00', False)], turns, CENTROIDS,
                             lambda spans: [voice(-0.9), voice(0.9), voice(-0.9)])
    assert [(p['text'], s) for p, s in passages] == [(' Hm. Right.', 'S01'), (' So.', 'S00')]
