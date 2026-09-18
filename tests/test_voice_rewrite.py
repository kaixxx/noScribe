"""The voice check writes the transcript a second time -- through the very writer
that wrote it the first time.

`on_segment` and `check_voices` are closures deep inside main.py's job routine,
out of reach of an ordinary test, and `on_segment` is a state machine that
upstream keeps evolving. What must hold however it evolves:

* writing the recorded passages again reproduces the document byte for byte
  wherever nothing moved -- overlap markers, inherited speakers, pause markers,
  time stamps and names included, in every output format;
* a passage that moved is written under its new speaker, and names still go to
  the speakers in the order they are first heard;
* a rewrite that fails half way puts back exactly what was there.

So this test lifts the closures out of main.py's source, as they are, and runs
them against a real AdvancedHTMLParser document. If one of them is renamed or
starts to need something the harness does not provide, the test fails loudly
rather than passing on nothing (test_the_harness_runs_the_real_closures).
"""
import ast
import datetime
import html
import random
import textwrap
import types
from pathlib import Path

import AdvancedHTMLParser
import pytest

from noScribe import utils, voice_check

MAIN = Path(__file__).resolve().parents[1] / 'noScribe' / 'main.py'
LIFTED = ('short_label', 'overlap_len', 'find_speaker', 'adjust_for_pause', 'on_segment', 'check_voices')
VOICES = {'SPEAKER_00': [1.0, 0.0, 0.0], 'SPEAKER_01': [0.0, 1.0, 0.0], 'SPEAKER_02': [0.0, 0.0, 1.0]}

HARNESS = '''
def build(self, job, d, main_body, diarization, tmp_audio_file):
    header_nodes = len(main_body.children)
    p = d.createElement('p')
    main_body.appendChild(p)
    speaker = speaker_disp = prev_speaker = ''
    last_auto_save = datetime.datetime.now()
    last_segment_end = last_timestamp_ms = 0
    first_segment = True
    voice_segments = []

    def save_doc():
        self.saved.append(d.asHTML())

{lifted}

    return on_segment, check_voices, voice_segments, (lambda: first_segment)
'''


def _lift():
    source = MAIN.read_text(encoding='utf-8')
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in LIFTED:
            found.setdefault(node.name, node)
    missing = [name for name in LIFTED if name not in found]
    assert not missing, f'main.py no longer defines {missing}: the rewrite is untested until this harness follows'
    lines = source.splitlines()
    bodies = [textwrap.indent(textwrap.dedent('\n'.join(lines[found[name].lineno - 1:found[name].end_lineno])), '    ')
              for name in LIFTED]
    return HARNESS.format(lifted='\n\n'.join(bodies))


class FakeApp:
    """What the closures ask of `self`."""
    cancel = False

    def __init__(self, voice_at):
        from noScribe.main import App
        self._apply_speaker_name = types.MethodType(App._apply_speaker_name, self)
        self.voice_at, self.logged, self.saved = voice_at, [], []

    def log(self, txt='', *args, **kwargs):
        self.logged.append(txt)

    def logn(self, txt='', *args, **kwargs):
        self.logged.append(f'{txt}\n')

    def set_progress(self, *args, **kwargs):
        pass

    def _run_voice_embeddings(self, tmp_audio_file, job, spans):
        return [self.voice_at((a + b) / 2) for a, b in spans]


def harness(diarization, voice_at, file_ext='html', names=(), overlapping=True):
    job = types.SimpleNamespace(
        start=0, timestamps=True, timestamp_interval=20000, timestamp_color='#78909C', pause=1,
        pause_marker='.', speaker_detection='auto', overlapping=overlapping, file_ext=file_ext,
        auto_save=False, speaker_names=list(names), speaker_name_map={}, speaker_centroids=VOICES)
    d = AdvancedHTMLParser.AdvancedHTMLParser()
    d.parseStr('<html><head></head><body></body></html>')
    main_body = d.createElement('div')
    d.body.appendChild(main_body)
    for title in ('title', 'subheader'):
        node = d.createElement('p')
        node.appendText(title)
        main_body.appendChild(node)
    app = FakeApp(voice_at)
    scope = {'datetime': datetime, 'html': html, 'utils': utils, 'voice_check': voice_check,
             't': lambda key, **kwargs: f'{key}{kwargs or ""}', 'duration': 3600.0, 'sampling_rate': 16000,
             'speech_chunks': [], 'is_voxtral': False}
    exec(compile(_lift(), str(MAIN), 'exec'), scope)
    on_segment, check_voices, voice_segments, first_segment = scope['build'](app, job, d, main_body, diarization, 'audio.wav')
    return types.SimpleNamespace(app=app, job=job, d=d, scope=scope, on_segment=on_segment,
                                 check_voices=check_voices, voice_segments=voice_segments,
                                 first_segment=first_segment)


def conversation(rnd):
    """Turns with gaps and talk-over, and segments that respect or straddle them."""
    turns, t, label = [], 0.0, 0
    for _ in range(rnd.randint(4, 9)):
        length = rnd.uniform(1.0, 9.0)
        turns.append({'start': round(t * 1000), 'end': round((t + length) * 1000), 'label': f'SPEAKER_0{label}'})
        if rnd.random() < 0.4:  # somebody talks into it
            a = t + rnd.uniform(0.2, length * 0.6)
            turns.append({'start': round(a * 1000), 'end': round((a + rnd.uniform(0.3, 0.9)) * 1000),
                          'label': f'SPEAKER_0{(label + 1) % 2}'})
        t += length + rnd.choice((0.0, 0.1, 0.4, 1.6, 2.5))
        label = (label + rnd.choice((1, 1, 2))) % 3
    turns.sort(key=lambda turn: turn['start'])
    segments, s = [], 0.0
    while s < t:
        words, w = [], s
        for k in range(rnd.randint(1, 9)):
            e = w + rnd.uniform(0.15, 0.6)
            words.append({'word': f' w{len(segments)}_{k}' + rnd.choice(('', '', ',', '.', '?')), 'start': round(w, 2), 'end': round(e, 2)})
            w = e + rnd.choice((0.0, 0.0, 0.05, 0.7))
        segments.append({'start': words[0]['start'], 'end': words[-1]['end'],
                         'text': ''.join(x['word'] for x in words), 'words': words})
        s = w + rnd.choice((0.0, 0.2, 1.4, 3.0))
    return turns, segments


def test_the_harness_runs_the_real_closures():
    source = _lift()
    for name in LIFTED:
        assert f'def {name}(' in source
    assert 'speaker_override' in source and 'voice_check.relabel' in source


@pytest.mark.parametrize('file_ext', ['html', 'vtt'])
@pytest.mark.parametrize('names', [(), ('Mona', 'Lena')])
def test_writing_the_passages_again_reproduces_the_document(monkeypatch, file_ext, names):
    """Forced through the rewrite with nothing moved: relabel's passages are the
    real ones, only the list of moves is made non-empty."""
    real = voice_check.relabel

    def one_fake_move(*args):
        passages, moves = real(*args)
        assert moves == []
        return passages, [(passages[0][0], 'S00', 'S00')]

    for seed in range(60):
        rnd = random.Random(seed)
        turns, segments = conversation(rnd)
        h = harness(turns, voice_at=None, file_ext=file_ext, names=names, overlapping=seed % 3 != 0)
        for segment in segments:
            h.on_segment(segment)
        written = {id(seg): speaker.lstrip('/') for seg, speaker, _ in h.voice_segments}
        spans_of = {(u[0]['start'], u[-1]['end']): written[id(seg)]
                    for seg, _, _ in h.voice_segments for u in voice_check.split_units(seg['words'])}
        h.app._run_voice_embeddings = lambda tmp, job, spans: [
            VOICES['SPEAKER_' + spans_of[(a, b)][1:]] for a, b in spans]
        before, names_before = h.d.asHTML(), dict(h.job.speaker_name_map)
        monkeypatch.setattr(voice_check, 'relabel', one_fake_move)
        h.check_voices()
        monkeypatch.setattr(voice_check, 'relabel', real)
        assert h.d.asHTML() == before, f'seed {seed}'
        assert h.job.speaker_name_map == names_before, f'seed {seed}'


def two_people():
    """S00 asks, S01 answers inside the same segment; the diarization only hears
    the first voice in it. Then S01 goes on."""
    turns = [{'start': 0, 'end': 4000, 'label': 'SPEAKER_00'}, {'start': 4200, 'end': 9000, 'label': 'SPEAKER_01'}]
    words = [{'word': ' Does', 'start': 0.2, 'end': 0.5}, {'word': ' it', 'start': 0.5, 'end': 0.7},
             {'word': ' help?', 'start': 0.7, 'end': 1.4}, {'word': ' Yes,', 'start': 2.0, 'end': 2.4},
             {'word': ' sure.', 'start': 2.4, 'end': 3.0}]
    first = {'start': 0.2, 'end': 3.0, 'text': ' Does it help? Yes, sure.', 'words': words}
    second = {'start': 4.5, 'end': 6.0, 'text': ' It really does.',
              'words': [{'word': ' It', 'start': 4.5, 'end': 4.8}, {'word': ' really', 'start': 4.8, 'end': 5.3},
                        {'word': ' does.', 'start': 5.3, 'end': 6.0}]}
    return turns, [first, second]


def paragraphs(h):
    return [node.textContent.strip() for node in h.d.body.children[0].children[2:] if node.textContent.strip()]


def test_a_moved_answer_is_written_under_its_speaker():
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 1.7 else 'SPEAKER_01'])
    for segment in segments:
        h.on_segment(segment)
    assert [text.split(':')[0] for text in paragraphs(h)] == ['S00', 'S01']
    assert 'Yes, sure.' in paragraphs(h)[0]
    h.check_voices()
    first, second = paragraphs(h)
    assert first.startswith('S00:') and first.endswith('Does it help?')
    assert second.startswith('S01:') and 'Yes, sure.' in second and second.endswith('It really does.')
    assert any('voice check: 00:00:02 S00 -> S01' in line for line in h.app.logged)


def test_names_follow_the_order_in_which_the_voices_are_first_heard():
    """The diarization put the opening under S01; the voice says it is S00. The
    first name belongs to whoever is heard first, so it has to change hands."""
    turns = [{'start': 0, 'end': 2000, 'label': 'SPEAKER_01'}, {'start': 2100, 'end': 6000, 'label': 'SPEAKER_01'}]
    opening = {'start': 0.1, 'end': 1.5, 'text': ' Welcome.', 'words': [{'word': ' Welcome.', 'start': 0.1, 'end': 1.5}]}
    reply = {'start': 2.2, 'end': 4.0, 'text': ' Thank you.', 'words': [{'word': ' Thank', 'start': 2.2, 'end': 2.8},
                                                                      {'word': ' you.', 'start': 2.8, 'end': 4.0}]}
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 2 else 'SPEAKER_01'], names=('Mona', 'Lena'))
    for segment in (opening, reply):
        h.on_segment(segment)
    assert paragraphs(h) == ['Mona: [00:00:00] Welcome. Thank you.']
    h.check_voices()
    assert paragraphs(h) == ['Mona: [00:00:00] Welcome.', 'Lena: [00:00:02] Thank you.']
    assert h.job.speaker_name_map == {'S00': 'Mona', 'S01': 'Lena'}


def test_a_rewrite_that_fails_half_way_puts_everything_back(monkeypatch):
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 1.7 else 'SPEAKER_01'], names=('Mona', 'Lena'))
    for segment in segments:
        h.on_segment(segment)
    before, names_before = h.d.asHTML(), dict(h.job.speaker_name_map)
    real = voice_check.relabel

    def poisoned(*args):
        passages, moves = real(*args)
        return passages[:-1] + [({'start': None, 'end': None, 'text': 'x', 'words': None}, 'S01')], moves

    monkeypatch.setattr(voice_check, 'relabel', poisoned)
    with pytest.raises(TypeError):
        h.check_voices()
    assert h.d.asHTML() == before
    assert h.job.speaker_name_map == names_before
    assert h.first_segment() is False  # so the job still saves the transcript
    assert h.app.saved == [before]      # and it was saved before anything was touched


def test_a_cancel_during_the_rewrite_leaves_the_transcript_whole():
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 1.7 else 'SPEAKER_01'])
    for segment in segments:
        h.on_segment(segment)
    before = h.d.asHTML()
    h.app._run_voice_embeddings = lambda tmp, job, spans: (
        setattr(h.app, 'cancel', True) or [h.app.voice_at((a + b) / 2) for a, b in spans])
    with pytest.raises(Exception, match='err_user_cancelation'):
        h.check_voices()
    assert h.d.asHTML() == before
