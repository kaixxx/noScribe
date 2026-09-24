"""The voice check writes the transcript a second time -- through the very writer
that wrote it the first time.

`on_segment` and `check_voices` are closures deep inside main.py's job routine,
out of reach of an ordinary test, and `on_segment` is a state machine that
upstream keeps evolving. What must hold however it evolves:

* writing the recorded passages again reproduces the document byte for byte
  wherever nothing moved -- overlap markers, inherited speakers, pause markers,
  time stamps and names included, in every output format;
* a passage that moved is written under its new speaker, and names still go to
  the speakers in the order they are first heard -- without a voice the check
  brings in taking the name of one the transcript already showed;
* a rewrite that fails half way puts back exactly what was there, and a check
  that fails, measures nothing or is stopped costs only itself, never the job.

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
import traceback
import types
from pathlib import Path

import AdvancedHTMLParser
import pytest

pytest.importorskip("tkinter")  # noScribe.main, for App._apply_speaker_name

from noScribe import utils, voice_check

MAIN = Path(__file__).resolve().parents[1] / 'noScribe' / 'main.py'
LIFTED = ('short_label', 'overlap_len', 'find_speaker', 'adjust_for_pause', 'on_segment', 'check_voices',
          'run_voice_check')
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

    return on_segment, check_voices, run_voice_check, voice_segments, (lambda: first_segment)
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
        self.log(f'{txt}\n')

    def set_progress(self, *args, **kwargs):
        pass

    def _run_voice_embeddings(self, tmp_audio_file, job, spans):
        return [self.voice_at((a + b) / 2) for a, b in spans]


@pytest.fixture(autouse=True)
def _no_switch_from_the_shell(monkeypatch):
    """NOSCRIBE_VOICE_CHECK=0 exported in the developer's shell turned the
    check off in every test that runs it."""
    monkeypatch.delenv('NOSCRIBE_VOICE_CHECK', raising=False)


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
             'speech_chunks': [], 'traceback': traceback, 'get_config': lambda key, default=None: default}
    exec(compile(_lift(), str(MAIN), 'exec'), scope)
    on_segment, check_voices, run_voice_check, voice_segments, first_segment = scope['build'](
        app, job, d, main_body, diarization, 'audio.wav')
    return types.SimpleNamespace(app=app, job=job, d=d, scope=scope, on_segment=on_segment,
                                 check_voices=check_voices, run_voice_check=run_voice_check,
                                 voice_segments=voice_segments,
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


def test_a_voice_the_check_brings_in_takes_no_name_from_a_voice_that_had_one():
    """Two names, and the diarization found two speakers -- plus a third cluster
    that has a centroid but never won a segment. The voice moves Lena's first
    passage to it. Handing out the names afresh in the order heard gave the new
    voice "Lena" and left the real Lena under a bare label, and the warning that
    there are more speakers than names never came: the rewrite is written quietly.
    The names the transcript showed stay where they were, the newcomer is
    numbered, and the warning is given because the overflow is new."""
    def seg(text, a, b):
        return {'start': a, 'end': b, 'text': f' {text}', 'words': [{'word': f' {text}', 'start': a, 'end': b}]}

    turns = [{'start': 0, 'end': 2000, 'label': 'SPEAKER_00'}, {'start': 2500, 'end': 9000, 'label': 'SPEAKER_01'}]
    voice_at = lambda t: VOICES['SPEAKER_00' if t < 2 else 'SPEAKER_02' if t < 4.5 else 'SPEAKER_01']
    h = harness(turns, voice_at=voice_at, names=('Mona', 'Lena'), overlapping=False)
    for segment in (seg('Hello.', 0.2, 1.5), seg('Who, me?', 3.0, 4.0), seg('It really does.', 5.0, 6.0)):
        h.on_segment(segment)
    assert [text.split(':')[0] for text in paragraphs(h)] == ['Mona', 'Lena']
    assert not any('warn_speaker_names_more_speakers' in line for line in h.app.logged)
    h.check_voices()
    assert [text.split(':')[0] for text in paragraphs(h)] == ['Mona', 'S02', 'Lena']
    assert h.job.speaker_name_map == {'S00': 'Mona', 'S01': 'Lena', 'S02': 'S02'}
    assert sum('warn_speaker_names_more_speakers' in line for line in h.app.logged) == 1


def test_a_rewrite_that_is_undone_leaves_no_warning_behind():
    """The overflow warning used to be given while the names were handed out,
    before the rewrite ran -- so a rewrite then stopped and put back left a red
    "more speakers than names" on screen for a transcript without a third."""
    def seg(text, a, b):
        return {'start': a, 'end': b, 'text': f' {text}', 'words': [{'word': f' {text}', 'start': a, 'end': b}]}

    turns = [{'start': 0, 'end': 2000, 'label': 'SPEAKER_00'}, {'start': 2500, 'end': 9000, 'label': 'SPEAKER_01'}]
    voice_at = lambda t: VOICES['SPEAKER_00' if t < 2 else 'SPEAKER_02' if t < 4.5 else 'SPEAKER_01']
    h = harness(turns, voice_at=voice_at, names=('Mona', 'Lena'), overlapping=False)
    for segment in (seg('Hello.', 0.2, 1.5), seg('Who, me?', 3.0, 4.0), seg('It really does.', 5.0, 6.0)):
        h.on_segment(segment)
    h.app._run_voice_embeddings = lambda tmp, job, spans: (
        setattr(h.app, 'cancel', True) or [voice_at((a + b) / 2) for a, b in spans])
    with pytest.raises(Exception, match='err_user_cancelation'):
        h.check_voices()
    assert h.job.speaker_name_map == {'S00': 'Mona', 'S01': 'Lena'}
    assert not any('warn_speaker_names_more_speakers' in line for line in h.app.logged)


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


def test_stop_during_the_check_ends_the_check_and_not_the_job():
    """Stop pressed while the check waits for its voices -- which can take
    minutes -- used to cancel the job, and a complete transcript was reported as
    canceled. The check is what Stop ends (the worker is terminated before the
    cancel is raised, see _run_pyannote_worker); the job goes on to finish with
    the transcript as diarized, and the screen says why nothing was checked."""
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 1.7 else 'SPEAKER_01'])
    for segment in segments:
        h.on_segment(segment)
    before = h.d.asHTML()

    def stopped(tmp, job, spans):
        h.app.cancel = True  # as the Stop button does, while the worker runs
        raise Exception('err_user_cancelation')
    h.app._run_voice_embeddings = stopped
    h.run_voice_check()  # does not raise: the job is not canceled
    assert h.d.asHTML() == before
    assert 'voice_check_canceled\n' in h.app.logged and 'voice_check_skipped\n' not in h.app.logged


def _pump(monkeypatch, messages, cancel_after=None):
    """Run the real _run_pyannote_worker against a worker that sends
    `messages`, in a thread instead of a spawned process; returns what was
    logged, as (text, where) pairs, and the error raised, if any."""
    import queue
    import threading
    import noScribe.main as m
    from noScribe import pyannote_mp_worker

    class _Proc:
        def __init__(self, target, args):
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.exitcode = 0

        def start(self):
            self.thread.start()

        def is_alive(self):
            return self.thread.is_alive()

        def join(self, timeout=None):
            self.thread.join(timeout)

        def terminate(self):
            pass

        def close(self):
            pass

    class _Queue(queue.Queue):
        def close(self):
            pass

        def join_thread(self):
            pass

    ctx = types.SimpleNamespace(Queue=_Queue, Process=_Proc)
    monkeypatch.setattr(m.mp, "get_context", lambda method: ctx)

    def entrypoint(args, q):
        for message in messages:
            q.put(message)
    monkeypatch.setattr(pyannote_mp_worker, "pyannote_proc_entrypoint", entrypoint)

    logged = []
    app = types.SimpleNamespace(cancel=False, _mp_proc=None, _mp_queue=None)
    app.log = lambda txt='', *a, where='both', **k: logged.append((txt, where))
    app.logn = lambda txt='', *a, where='both', **k: logged.append((f'{txt}\n', where))
    app.logr = lambda txt='', *a, where='both', **k: logged.append((txt, where))
    app.set_progress = lambda *a, **k: None
    job = types.SimpleNamespace(speaker_detection='auto')
    try:
        m.App._run_pyannote_worker(app, 'audio.wav', job, {}, errors_to='file')
        return logged, None
    except Exception as err:
        return logged, err


def test_a_message_after_a_progress_line_starts_a_line_of_its_own(monkeypatch):
    """logr writes the progress without a line end, and what came after it was
    glued on: "voice_check: 100%PyAnnote error: ..." in the log file, and the
    first diarization segment after "discrete_diarization: 100%"."""
    progress = {"type": "progress", "step": "voice_check", "pct": 100}
    for last in ({"type": "result", "ok": True, "segments": []},
                 {"type": "result", "ok": False, "error": "no span could be embedded"},
                 {"type": "log", "level": "warn", "msg": "2 of 3 spans failed"}):
        messages = [progress, last] + ([] if last["type"] == "result" else
                                       [{"type": "result", "ok": True}])
        logged, _ = _pump(monkeypatch, messages)
        at = [text for text, _ in logged].index('voice_check: 100%')
        assert logged[at + 1] == ('\n', 'both'), logged


def test_stop_before_a_check_with_nothing_to_do_says_nothing():
    """With one voice there is nothing to check, so there is nothing a Stop
    could have ended either -- "voice check stopped" would be news of nothing."""
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00'])
    h.job.speaker_centroids = {'SPEAKER_00': VOICES['SPEAKER_00']}
    for segment in segments:
        h.on_segment(segment)
    h.app.cancel = True
    h.run_voice_check()
    assert not any(line.startswith('voice_check') for line in h.app.logged)


def test_stop_just_before_the_check_skips_it_without_starting_it():
    """The transcription came back complete, and Stop arrived before the check
    began: the same as a Stop during it, and no worker is started for nothing."""
    turns, segments = two_people()
    h = harness(turns, voice_at=lambda t: VOICES['SPEAKER_00' if t < 1.7 else 'SPEAKER_01'])
    for segment in segments:
        h.on_segment(segment)
    before = h.d.asHTML()
    h.app.cancel = True

    started = []
    h.app._run_voice_embeddings = lambda tmp, job, spans: started.append(spans) or []
    h.run_voice_check()
    assert started == []
    assert h.d.asHTML() == before
    assert 'voice_check_canceled\n' in h.app.logged


def test_a_check_that_measured_no_voice_says_so():
    """When the worker can measure no voice at all (pyannote's private
    `_embedding` gone, another sample rate, every span rejected), it now fails
    the call (tests/test_voice_embeddings.py) -- before, it handed back a list
    of None and the screen said "0 passage(s) reassigned", as if every speaker
    had been checked and found right."""
    turns, segments = two_people()
    h = harness(turns, voice_at=None)
    for segment in segments:
        h.on_segment(segment)
    before = h.d.asHTML()

    def worker_failed(tmp, job, spans):
        raise Exception('RuntimeError: no span could be embedded (device_mps)')
    h.app._run_voice_embeddings = worker_failed
    h.run_voice_check()
    assert h.d.asHTML() == before
    assert 'voice_check_skipped\n' in h.app.logged
    assert not any(line.startswith('voice_check_done') for line in h.app.logged)
    assert any('no span could be embedded' in line for line in h.app.logged)  # the reason, for the log file
    assert any('Traceback (most recent call last)' in line for line in h.app.logged)  # and where it came from


def test_a_job_stopped_during_the_voice_check_opens_no_editor(monkeypatch):
    """Stop during the voice check lets the job finish -- its transcript is
    complete -- and a finished single html job opens the editor. Whoever
    pressed Stop, for the whole queue or for this job, did not ask for that.
    The queue worker runs for real; only the job's own processing is stubbed."""
    import noScribe.main as m
    from noScribe.main import App, JobStatus
    monkeypatch.setitem(m.config, 'auto_edit_transcript', 'True')   # not the user's own

    class Job:
        def __init__(self):
            self.status, self.file_ext, self.transcript_file = JobStatus.WAITING, 'html', 'x.html'
            self.audio_file, self.error_message = 'a.wav', None

        def set_finished(self):
            self.status = JobStatus.FINISHED

        def set_canceled(self, message=None):
            self.status = JobStatus.CANCELED

        def set_error(self, message, tb=None):
            self.status = JobStatus.ERROR

    class Queue:
        def __init__(self, jobs):
            self.jobs = jobs

        def get_queue_summary(self):
            return {'total': len(self.jobs), 'finished': 0, 'errors': 0, 'canceled': 0}

        def get_waiting_jobs(self):
            return [job for job in self.jobs if job.status == JobStatus.WAITING]

        def has_pending_jobs(self):
            return bool(self.get_waiting_jobs())

        def get_next_waiting_job(self):
            waiting = self.get_waiting_jobs()
            return waiting[0] if waiting else None

    def run(stop):
        app = types.SimpleNamespace(_headless=False, _cancel_job_only=False, cancel=False, opened=[])
        app.queue = Queue([Job()])
        app.logn = app.set_progress = app.update_queue_table = lambda *a, **k: None
        app.launch_editor = app.opened.append

        def process(job):
            job.status = JobStatus.TRANSCRIPTION
            stop(app, job)  # during the voice check, which run_voice_check ends quietly
        app._process_single_job = process
        App.transcription_worker(app)
        return app.opened, app.queue.jobs[0].status

    def stop_queue(app, job):  # as on_queue_stop does
        job.status, app.cancel, app._cancel_job_only = JobStatus.CANCELING, True, False

    def stop_job(app, job):    # as the cancel button on the job's row does
        job.status, app.cancel, app._cancel_job_only = JobStatus.CANCELING, True, True

    assert run(lambda app, job: None) == (['x.html'], JobStatus.FINISHED)
    assert run(stop_queue) == ([], JobStatus.FINISHED)
    assert run(stop_job) == ([], JobStatus.FINISHED)
