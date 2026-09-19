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

    def __init__(self, reject_longer_than=None):
        self.lengths, self.reject = [], reject_longer_than

    def __call__(self, batch):
        assert batch.shape[:2] == (1, 1)  # one mono crop at a time
        self.lengths.append(batch.shape[2])
        if self.reject and batch.shape[2] > self.reject:
            raise RuntimeError("too long for this fake")
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


def test_a_span_beyond_the_file_gets_none_and_the_model_is_not_asked():
    model = FakeModel()
    out, _ = embed([[12.0, 13.0]], model)
    assert out == [None] and model.lengths == []


def test_without_the_model_or_at_another_rate_nothing_is_embedded():
    """The crops bypass pyannote's own resampling, so they are only valid at the
    model's rate; and `_embedding` is not a public attribute, so it may be gone."""
    other = FakeModel(); other.sample_rate = 8000
    assert embed([[0.0, 1.0]], other)[0] == [None]
    q = Queue()
    assert worker._embed_spans(types.SimpleNamespace(), torch.zeros(1, RATE), RATE, [[0.0, 1.0]], q) == [None]


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
