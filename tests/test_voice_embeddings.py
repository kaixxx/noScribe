"""The diarization worker's half of the voice check: the speaker centroids it
hands back with the diarization, and the one-embedding-per-span mode.

Neither needs a model here -- the embedding model is a callable, and a fake one
shows what the worker asks of it."""
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from noScribe import pyannote_mp_worker as worker

RATE = 16000


class FakeModel:
    sample_rate = RATE

    def __init__(self, reject_longer_than=None, answers=None):
        self.lengths, self.reject, self.answers = [], reject_longer_than, list(answers or [])

    def __call__(self, batch):
        assert batch.shape[:2] == (1, 1)  # one mono crop at a time
        self.lengths.append(batch.shape[2])
        if self.reject and batch.shape[2] > self.reject:
            raise RuntimeError("too long for this fake")
        if self.answers:
            return np.array([self.answers.pop(0)], dtype="float32")
        return np.ones((1, 4), dtype="float32")


class Queue(list):
    put = list.append


def embed(spans, model, seconds=10):
    q = Queue()
    pipeline = types.SimpleNamespace(_embedding=model)
    return worker._embed_spans(pipeline, torch.zeros(1, seconds * RATE), RATE, spans, q), q


def test_one_answer_per_span_and_short_spans_are_widened():
    model = FakeModel()
    out, q = embed([[1.0, 3.0], [5.0, 5.1], [0.0, 0.1], [9.95, 10.0]], model)
    assert [v is not None for v in out] == [True] * 4
    shortest = int(worker.EMBED_MIN_S * RATE)
    # as asked for; widened evenly; widened against the start of the file, where
    # only the right half fits; and the same against its end
    assert model.lengths[:2] == [2 * RATE, shortest]
    assert shortest // 2 <= model.lengths[2] < shortest and shortest // 2 <= model.lengths[3] < shortest
    assert q[-1] == {"type": "progress", "step": "voice_check", "pct": 100}


def test_a_span_the_model_rejects_costs_only_itself():
    out, _ = embed([[0.0, 1.0], [2.0, 6.0], [7.0, 8.0]], FakeModel(reject_longer_than=2 * RATE))
    assert [v is not None for v in out] == [True, False, True]


def test_rejected_spans_are_logged_once_with_the_first_reason():
    """The rejections used to vanish without a trace, so a check that measured
    nothing could not tell why. One line, however many spans fail: on a long
    recording the same failure tends to repeat thousands of times."""
    out, q = embed([[0.0, 1.0], [2.0, 6.0], [6.0, 9.0]], FakeModel(reject_longer_than=2 * RATE))
    logs = [m for m in q if m["type"] == "log"]
    assert len(logs) == 1 and logs[0]["level"] == "warn"
    assert "2 of 3" in logs[0]["msg"] and "RuntimeError: too long for this fake" in logs[0]["msg"]
    _, q = embed([[0.0, 1.0]], FakeModel())
    assert not [m for m in q if m["type"] == "log"]


def test_a_vector_without_a_direction_counts_as_a_failure():
    """NaN became None without a trace, and an all-zero vector counted as a
    voice: its cosine with every centroid is undefined, so the check kept each
    passage and said "0 reassigned" as though it had confirmed them."""
    nan, zero, good = [np.nan] * 4, [0.0] * 4, [1.0] * 4
    out, q = embed([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], FakeModel(answers=[nan, zero, good]))
    assert out[:2] == [None, None] and out[2] == good
    logs = [m for m in q if m["type"] == "log"]
    assert len(logs) == 1 and "2 of 3" in logs[0]["msg"]
    with pytest.raises(RuntimeError, match="no span could be embedded.*NaN or zeros"):
        embed([[0.0, 1.0], [2.0, 3.0]], FakeModel(answers=[zero, zero]))


def test_a_span_beyond_the_file_gets_none_and_the_model_is_not_asked():
    model = FakeModel()
    out, _ = embed([[0.0, 1.0], [12.0, 13.0]], model)
    assert out[1] is None and model.lengths == [RATE]


def test_without_the_model_or_at_another_rate_the_check_does_not_run():
    """The crops bypass pyannote's own resampling, so they are only valid at the
    model's rate; and `_embedding` is not a public attribute, so it may be gone.
    Either way not one voice is measured, and a list of None read on screen
    like a check that found every speaker right."""
    other = FakeModel(); other.sample_rate = 8000
    with pytest.raises(RuntimeError, match="8000 Hz"):
        embed([[0.0, 1.0]], other)
    with pytest.raises(RuntimeError, match="no embedding model"):
        worker._embed_spans(types.SimpleNamespace(), torch.zeros(1, RATE), RATE, [[0.0, 1.0]], Queue())


def test_a_check_where_every_span_fails_does_not_run_either():
    with pytest.raises(RuntimeError, match="no span could be embedded.*too long for this fake"):
        embed([[0.0, 3.0], [4.0, 8.0]], FakeModel(reject_longer_than=2 * RATE))


def test_centroids_skip_the_rows_that_are_none():
    """More labels than clusters pads the matrix with zero rows, and a cluster can
    come back as NaN; a cosine against either would be noise."""
    matrix = np.array([[0.5, 0.5], [0.0, 0.0], [np.nan, 1.0]])
    labels = types.SimpleNamespace(labels=lambda: ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
    found = worker._centroids(types.SimpleNamespace(speaker_embeddings=matrix, speaker_diarization=labels))
    assert found == {"SPEAKER_00": [0.5, 0.5]}
    assert worker._centroids(types.SimpleNamespace()) == {}


def test_memory_that_cannot_be_asked_about_counts_as_none(monkeypatch):
    """os.sysconf does not exist on Windows; there the embed call stays where
    the diarization ran."""
    monkeypatch.delattr(worker.os, "sysconf", raising=False)
    assert worker._ram_gb() == 0
