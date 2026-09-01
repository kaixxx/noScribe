"""Tests for cutting a transcript segment where the diarization speaker changes.

A segment is the unit a speaker is assigned to, so one that spans a turn hands
the whole text to whoever overlaps it most and the shorter half is lost. The
regression that motivated this is pinned below with the real timings from the
run that produced it: the segment builder had merged the turn-final answer
"Ja, unbedingt." into the question before it, and the merged cue went to the
questioner on 40.4% overlap against 39.3%.
"""
import pytest

from noScribe.main import split_at_speaker_change


# The diarization of the first 11 s of the recording (pyannote, verbatim from
# the run's log), and the word stamps forced alignment produced for it.
DIARIZATION = [
    {"start": 30, "end": 2494, "label": "SPEAKER_01"},
    {"start": 3119, "end": 8670, "label": "SPEAKER_00"},
    {"start": 9278, "end": 10830, "label": "SPEAKER_01"},
    {"start": 11387, "end": 15319, "label": "SPEAKER_01"},
]


def _speaker_of(start_ms, end_ms):
    """The caller's assignment rule: best overlap wins, shortest on a tie."""
    best, spkr, seg_len = 0.0, '', 0
    for seg in DIARIZATION:
        if end_ms < seg["start"]:
            break
        if start_ms > seg["end"] or end_ms <= start_ms:
            continue
        overlap = (min(seg["end"], end_ms) - max(seg["start"], start_ms) + 1)
        share = overlap / (end_ms - start_ms)
        current_len = seg["end"] - seg["start"]
        if share > best or (share == best and current_len < seg_len):
            best, spkr, seg_len = share, f'S{seg["label"][8:]}', current_len
    return spkr


def _words(*spec):
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def _texts(pieces):
    return [p["text"].strip() for p in pieces]


# --------------------------------------------------------------------------- #
# The regression
# --------------------------------------------------------------------------- #
def test_turn_final_answer_is_cut_off_the_question():
    """"der digitalen Welt? Ja, unbedingt." is two speakers, not one."""
    seg = {"start": 7.47, "end": 10.45, "text": " der digitalen Welt? Ja, unbedingt.",
           "words": _words(("der", 7.47, 7.57), ("digitalen", 7.63, 8.21),
                           ("Welt?", 8.27, 8.59), ("Ja,", 9.35, 9.47),
                           ("unbedingt.", 9.91, 10.45))}
    pieces = split_at_speaker_change(seg, _speaker_of)
    assert _texts(pieces) == ["der digitalen Welt?", "Ja, unbedingt."]
    assert _speaker_of(round(pieces[0]["start"] * 1000), round(pieces[0]["end"] * 1000)) == "S00"
    assert _speaker_of(round(pieces[1]["start"] * 1000), round(pieces[1]["end"] * 1000)) == "S01"


def test_the_uncut_segment_would_go_to_the_wrong_speaker():
    """Without the cut the answer is swallowed by the question -- narrowly."""
    assert _speaker_of(7470, 10450) == "S00"   # 40.4% against 39.3%


# --------------------------------------------------------------------------- #
# What must not be cut
# --------------------------------------------------------------------------- #
def test_one_speaker_stays_one_segment():
    seg = {"start": 3.2, "end": 8.5, "text": " Herzlich willkommen. Willst du?",
           "words": _words(("Herzlich", 3.2, 4.0), ("willkommen.", 4.1, 4.6),
                           ("Willst", 4.84, 5.14), ("du?", 5.18, 8.5))}
    assert split_at_speaker_change(seg, _speaker_of) == [seg]


def test_a_single_sentence_is_never_cut():
    """Speakers change between sentences; mid-sentence there is nowhere to cut."""
    seg = {"start": 3.2, "end": 10.4, "text": " Herzlich willkommen",
           "words": _words(("Herzlich", 3.2, 4.0), ("willkommen", 9.9, 10.4))}
    assert split_at_speaker_change(seg, _speaker_of) == [seg]


def test_a_gap_with_no_diarization_does_not_cut():
    """'' means no overlap, not a new speaker -- the caller keeps the current one."""
    seg = {"start": 3.2, "end": 4.6, "text": " Hm. Ja.",
           "words": _words(("Hm.", 3.2, 3.4), ("Ja.", 4.4, 4.6))}
    assert split_at_speaker_change(seg, lambda s, e: "" if s > 4000 else "S00") == [seg]


@pytest.mark.parametrize("seg", [
    {"start": 0.0, "end": 1.0, "text": " Ja.", "words": None},
    {"start": 0.0, "end": 1.0, "text": " Ja."},
    {"start": 0.0, "end": 1.0, "text": " Ja.", "words": []},
])
def test_segments_without_word_stamps_pass_through(seg):
    assert split_at_speaker_change(seg, _speaker_of) == [seg]


def test_incomplete_word_stamps_pass_through():
    """A missing stamp gives no basis to cut on -- leave the segment alone."""
    seg = {"start": 7.47, "end": 10.45, "text": " Welt? Ja.",
           "words": [{"word": "Welt?", "start": 7.47, "end": 8.59},
                     {"word": "Ja.", "start": None, "end": None}]}
    assert split_at_speaker_change(seg, _speaker_of) == [seg]


# --------------------------------------------------------------------------- #
# Text reassembly
# --------------------------------------------------------------------------- #
def test_whisper_style_words_keep_their_own_spacing():
    """faster-whisper words carry a leading space; another engine's may be bare tokens.
    Both have to come back out as normal text."""
    seg = {"start": 8.27, "end": 10.45, "text": " Welt? Ja, unbedingt.",
           "words": _words((" Welt?", 8.27, 8.59), (" Ja,", 9.35, 9.47),
                           (" unbedingt.", 9.91, 10.45))}
    pieces = split_at_speaker_change(seg, _speaker_of)
    assert [p["text"] for p in pieces] == [" Welt?", " Ja, unbedingt."]


def test_pieces_carry_their_own_words_and_bounds():
    seg = {"start": 7.47, "end": 10.45, "text": " der digitalen Welt? Ja, unbedingt.",
           "words": _words(("der", 7.47, 7.57), ("digitalen", 7.63, 8.21),
                           ("Welt?", 8.27, 8.59), ("Ja,", 9.35, 9.47),
                           ("unbedingt.", 9.91, 10.45))}
    first, second = split_at_speaker_change(seg, _speaker_of)
    assert (first["start"], first["end"]) == (7.47, 8.59)
    assert (second["start"], second["end"]) == (9.35, 10.45)
    assert len(first["words"]) == 3 and len(second["words"]) == 2


def test_runs_of_the_same_speaker_stay_together():
    """Three sentences, two speakers: two pieces, not three."""
    seg = {"start": 3.2, "end": 10.45, "text": " A. B. C.",
           "words": _words(("Willkommen.", 3.2, 4.0), ("Welt?", 8.27, 8.59),
                           ("Unbedingt.", 9.91, 10.45))}
    pieces = split_at_speaker_change(seg, _speaker_of)
    assert _texts(pieces) == ["Willkommen. Welt?", "Unbedingt."]
