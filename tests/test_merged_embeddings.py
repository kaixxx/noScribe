"""Unit tests for _Voxtral._merged_embeddings, the audio/text prompt merge.

The prompt is almost entirely [AUDIO] placeholders (375 tokens per 30 s), and
the projected audio embeddings have to land on exactly those positions. The
engine merges them itself rather than calling mlx_voxtral's private
`_merge_input_embeddings` (equivalent since 0.0.6, on the batch of 1 production
uses) -- so these run in CI, where the model-gated smoke test does not.

Driven with a stub model: no weights, no real Voxtral, but real mlx arrays,
because the dtype promotion these pin is an mlx behaviour.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from noScribe.voxtral_engine import _Voxtral

AUDIO_ID = 24
HIDDEN = 4


class _Config:
    audio_token_id = AUDIO_ID


class _StubModel:
    """Text embeddings in bf16 (as the real embed_tokens returns), audio
    embeddings in float32 (as the real projector returns)."""

    config = _Config()

    def __init__(self, audio_embeds):
        self.audio_embeds = audio_embeds
        self.seen_features = None

    def embed_tokens(self, input_ids):
        ids = np.array(input_ids).astype(np.float32)
        rows = np.repeat(ids[..., None], HIDDEN, axis=-1) + np.arange(HIDDEN)
        return mx.array(rows).astype(mx.bfloat16)

    def get_audio_embeds(self, features):
        self.seen_features = features
        return self.audio_embeds


def _engine(audio_embeds):
    v = _Voxtral.__new__(_Voxtral)  # no __init__: only the merge is exercised
    v._mx = mx
    v.model = _StubModel(audio_embeds)
    return v


def _audio(n, batch=1):
    """Values that bf16 cannot represent exactly, so a lost promotion shows."""
    vals = 0.1 + np.arange(batch * n * HIDDEN, dtype=np.float32) * 0.017
    return mx.array(vals.reshape(batch, n, HIDDEN))


def test_audio_lands_on_the_placeholders_and_text_is_untouched():
    ids = mx.array([[1, AUDIO_ID, AUDIO_ID, AUDIO_ID, 7]])
    audio = _audio(3)
    out = _engine(audio)._merged_embeddings({"input_ids": ids, "input_features": "mel"})
    assert out.shape == (1, 5, HIDDEN)
    assert mx.all(out[0, 1:4] == audio[0]), "audio embeddings not placed verbatim"
    text = _StubModel(audio).embed_tokens(ids).astype(out.dtype)
    assert mx.all(out[0, 0] == text[0, 0]) and mx.all(out[0, 4] == text[0, 4])


def test_non_contiguous_placeholders_keep_their_order():
    ids = mx.array([[AUDIO_ID, 5, AUDIO_ID, 6, AUDIO_ID]])
    audio = _audio(3)
    out = _engine(audio)._merged_embeddings({"input_ids": ids, "input_features": "mel"})
    for k, pos in enumerate((0, 2, 4)):
        assert mx.all(out[0, pos] == audio[0, k]), f"placeholder {k} got the wrong row"


def test_the_result_is_promoted_to_the_audio_dtype():
    """embed_tokens is bf16, the projector is float32. Scattering into the bf16
    array would round every audio embedding away -- silently."""
    ids = mx.array([[1, AUDIO_ID, AUDIO_ID, 7]])
    audio = _audio(2)
    out = _engine(audio)._merged_embeddings({"input_ids": ids, "input_features": "mel"})
    assert out.dtype == mx.float32
    assert mx.all(out[0, 1:3] == audio[0]), "audio embeddings lost precision"
    assert not mx.all(out[0, 1:3] == audio[0].astype(mx.bfloat16).astype(mx.float32)), \
        "test value survives a bf16 round-trip, so it cannot detect the bug"


def test_several_batch_rows_each_get_their_own_audio():
    ids = mx.array([[AUDIO_ID, AUDIO_ID, 3], [4, AUDIO_ID, AUDIO_ID]])
    audio = _audio(2, batch=2)
    out = _engine(audio)._merged_embeddings({"input_ids": ids, "input_features": "mel"})
    assert mx.all(out[0, 0:2] == audio[0]) and mx.all(out[1, 1:3] == audio[1])


def test_one_audio_stream_is_shared_by_every_batch_row():
    """get_audio_embeds returns [1, n, hidden] whatever went in, so a multi-row
    prompt shares it. Indexing it per row reads past the end, and mlx answers
    that with zeros instead of raising -- audio silently replaced by nothing."""
    ids = mx.array([[AUDIO_ID, AUDIO_ID, 3], [4, AUDIO_ID, AUDIO_ID]])
    audio = _audio(2)  # batch 1, as the encoder really returns it
    out = _engine(audio)._merged_embeddings({"input_ids": ids, "input_features": "mel"})
    assert mx.all(out[0, 0:2] == audio[0]) and mx.all(out[1, 1:3] == audio[0])


def test_an_impossible_stream_count_is_refused_rather_than_zero_filled():
    """Neither one shared stream nor one per row. Indexing past the end of an
    mlx array returns zeros rather than raising, so the prompt would silently
    lose its audio."""
    ids = mx.array([[AUDIO_ID, AUDIO_ID], [AUDIO_ID, AUDIO_ID],
                    [AUDIO_ID, AUDIO_ID]])
    with pytest.raises(ValueError, match="audio streams"):
        _engine(_audio(2, batch=2))._merged_embeddings(
            {"input_ids": ids, "input_features": "mel"})


def test_a_prompt_without_audio_is_just_the_text_embeddings():
    ids = mx.array([[1, 2, 3]])
    engine = _engine(_audio(0))
    out = engine._merged_embeddings({"input_ids": ids, "input_features": None})
    assert out.dtype == mx.bfloat16
    assert engine.model.seen_features is None, "encoder ran for a prompt with no audio"


def test_a_count_mismatch_is_refused_rather_than_scattered():
    """Placing N embeddings on M placeholders would corrupt the prompt quietly."""
    ids = mx.array([[AUDIO_ID, AUDIO_ID, AUDIO_ID]])
    with pytest.raises(ValueError, match="placeholder"):
        _engine(_audio(2))._merged_embeddings({"input_ids": ids, "input_features": "mel"})


def test_the_encoder_sees_the_features_it_was_given():
    ids = mx.array([[AUDIO_ID]])
    engine = _engine(_audio(1))
    engine._merged_embeddings({"input_ids": ids, "input_features": "the-mel"})
    assert engine.model.seen_features == "the-mel"
