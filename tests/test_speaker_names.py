"""Regression tests for the speaker-names feature (PR #323).

Covers name parsing/sanitization and the diarization-label -> user-name mapping,
including the identity-vs-display separation that keeps two speakers with the
same name in separate paragraphs.
"""
from types import SimpleNamespace

import noScribe.main as m
from noScribe.main import parse_speaker_names


def test_parse_splits_and_trims():
    assert parse_speaker_names("Mona, Lena") == ["Mona", "Lena"]
    assert parse_speaker_names("Mona; Lena") == ["Mona", "Lena"]
    assert parse_speaker_names(["Mona", "Lena"]) == ["Mona", "Lena"]


def test_parse_none_and_blank():
    assert parse_speaker_names(None) == []
    assert parse_speaker_names("") == []
    assert parse_speaker_names("   ") == []
    assert parse_speaker_names(",;") == []


def test_parse_sanitizes_anchor_breaking_chars():
    # underscore -> space (underscore is the anchor field separator)
    assert parse_speaker_names("Anna_Lena") == ["Anna Lena"]
    # <, >, ", & are removed (they break the HTML anchor attribute)
    assert parse_speaker_names('<Boss>, AT&T, "Q"') == ["Boss", "ATT", "Q"]
    # colon is removed (it is the "Name: text" label separator)
    assert parse_speaker_names("Dr: Smith") == ["Dr Smith"]


def _stub_app():
    """Minimal App-like object for calling _apply_speaker_name unbound.
    All the App contributes is the log sink for the overflow warning."""
    logs = []
    return SimpleNamespace(
        logn=lambda *a, **k: logs.append(a),
        _logs=logs,
    )


def _job(names):
    """A job carries both the names and the mapping built from them."""
    return SimpleNamespace(
        speaker_names=parse_speaker_names(names),
        speaker_name_map={},
    )


def test_real_job_carries_the_mapping_state():
    """The stubs above are SimpleNamespaces, so they would not notice if the
    real job stopped providing the mapping attributes _apply_speaker_name uses.
    Two jobs must also start out with separate maps."""
    job_a = m.create_transcription_job(speaker_names="Mona, Lena")
    job_b = m.create_transcription_job()
    assert job_a.speaker_name_map == {} and job_b.speaker_name_map == {}
    m.App._apply_speaker_name(_stub_app(), "S01", job_a)
    assert job_b.speaker_name_map == {}


def test_a_repeated_job_rebuilds_the_mapping():
    """The queue's repeat button re-runs the same job object. Diarization runs
    again from scratch and may hand out different labels, so the mapping must
    not survive into the second run -- otherwise the next label first seen gets
    names[len(old_map)] instead of names[0]."""
    app = _stub_app()
    job = m.create_transcription_job(speaker_names="Mona, Lena")

    job.set_running()
    assert m.App._apply_speaker_name(app, "S02", job) == "Mona"

    job.set_canceled("stopped")   # the state the repeat button acts on
    job.set_running()             # ... and what restarting it does
    assert job.speaker_name_map == {}
    # the first speaker heard on the second run gets the first name again
    assert m.App._apply_speaker_name(app, "S00", job) == "Mona"
    assert m.App._apply_speaker_name(app, "S01", job) == "Lena"


def test_maps_in_first_appearance_order():
    app, job = _stub_app(), _job("Mona, Lena")
    # first label heard gets the first name, regardless of S00/S01 numbering
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"
    assert m.App._apply_speaker_name(app, "S02", job) == "Lena"
    # stable on repeat
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"


def test_overlap_prefix_preserved():
    app, job = _stub_app(), _job("Mona, Lena")
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"
    # an overlapping turn keeps the // marker on the mapped name
    assert m.App._apply_speaker_name(app, "//S01", job) == "//Mona"


def test_overflow_is_numbered_by_appearance_and_warns_once():
    app, job = _stub_app(), _job("OnlyOne")
    assert m.App._apply_speaker_name(app, "S01", job) == "OnlyOne"
    # more speakers than names -> the extra speakers get their place in the order
    # of appearance, not pyannote's cluster number
    assert m.App._apply_speaker_name(app, "S03", job) == "S01"
    assert m.App._apply_speaker_name(app, "S00", job) == "S02"
    # warned exactly once across all overflow speakers
    assert sum(1 for a in app._logs if a) == 1


def test_without_names_speakers_are_numbered_in_the_order_they_appear():
    """pyannote's labels are cluster numbers: whoever opens the recording is as
    likely S01 as S00. Someone who transcribes interviews knows who speaks first
    and renames the speakers afterwards -- and had to look up, every time, which
    of them S01 happened to be this time. Now the first voice is always S00."""
    app, job = _stub_app(), _job("")
    assert m.App._apply_speaker_name(app, "S01", job) == "S00"
    assert m.App._apply_speaker_name(app, "S02", job) == "S01"
    assert m.App._apply_speaker_name(app, "S00", job) == "S02"
    # stable from then on, overlap marker kept, and nothing for the user to read
    assert m.App._apply_speaker_name(app, "S01", job) == "S00"
    assert m.App._apply_speaker_name(app, "//S02", job) == "//S01"
    assert app._logs == []


def test_duplicate_names_map_to_distinct_keys():
    """Two speakers given the same name still occupy distinct map entries, so the
    on_segment paragraph logic (which compares the raw labels) can keep them
    apart even though their display name is identical."""
    app, job = _stub_app(), _job("Anna, Anna")
    assert m.App._apply_speaker_name(app, "S01", job) == "Anna"
    assert m.App._apply_speaker_name(app, "S02", job) == "Anna"
    assert set(job.speaker_name_map) == {"S01", "S02"}


def test_a_speaker_first_heard_talking_over_someone_is_numbered_too():
    app, job = _stub_app(), _job("")
    assert m.App._apply_speaker_name(app, "S01", job) == "S00"
    assert m.App._apply_speaker_name(app, "//S00", job) == "//S01"
    assert m.App._apply_speaker_name(app, "S00", job) == "S01"


def test_no_speaker_stays_no_speaker():
    """find_speaker returns '' where no diarization turn overlaps; that is not a
    speaker and must not use up a number."""
    app, job = _stub_app(), _job("")
    assert m.App._apply_speaker_name(app, "", job) == ""
    assert m.App._apply_speaker_name(app, "S01", job) == "S00"


def test_the_log_file_says_which_of_pyannotes_labels_a_number_stands_for():
    """The diarization turns in the log file carry pyannote's labels; a transcript
    "S01" looks like "SPEAKER_01" without being it, so the log says which is which."""
    app, job = _stub_app(), _job("Mona")
    m.App._apply_speaker_name(app, "S01", job)
    m.App._apply_speaker_name(app, "S00", job)
    m.App._apply_speaker_name(app, "//S01", job)
    assert m.App._speaker_key(job) == "Mona = SPEAKER_01, S01 = SPEAKER_00"


def test_a_number_somebody_was_given_as_a_name_is_stepped_over():
    app, job = _stub_app(), _job("S01")
    assert m.App._apply_speaker_name(app, "S02", job) == "S01"   # the name
    assert m.App._apply_speaker_name(app, "S00", job) == "S02"   # not a second "S01"


def test_a_job_without_a_list_of_names_is_numbered_too():
    app, job = _stub_app(), _job("")
    job.speaker_names = None
    assert m.App._apply_speaker_name(app, "S01", job) == "S00"


def test_the_key_is_logged_wherever_the_transcript_is_saved():
    """The key used to be logged only on success, while a canceled or failed job
    still saves its partial transcript under the same numbers. It belongs where
    the transcript is saved -- the `finally` of the transcription step -- and
    only there, so a finished job does not log it twice. (That step is inline in
    the job routine, out of reach of a call; the source is what can be checked.)"""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(m.__file__).read_text(encoding='utf-8'))

    def calls(node, name):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                and (getattr(n.func, 'attr', None) == name or getattr(n.func, 'id', None) == name)]

    key_calls = calls(tree, '_speaker_key')
    assert len(key_calls) == 1
    saving = [stmt for node in ast.walk(tree) if isinstance(node, ast.Try)
              for stmt in node.finalbody if calls(stmt, 'save_doc')]
    assert saving and any(key_calls[0] in calls(stmt, '_speaker_key') for stmt in saving)
