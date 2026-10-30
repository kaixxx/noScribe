"""Unit tests for _Voxtral._consume_tokens, the token-stream bookkeeping of the
fast greedy path.

_fast_generate delegates generation to the maintained mlx_lm.generate_step and
keeps only the consumption logic here: stop at the first stop token. It
deliberately does NOT replicate generate_stream's 10-identical-token backstop --
a repetition loop must run so the engine's text-level _looks_degenerate detector
can catch it and retry. These tests drive that logic with plain int streams --
no weights, no mlx, no real Voxtral.
"""
from noScribe.voxtral_engine import _Voxtral


def _consume(stream):
    v = _Voxtral.__new__(_Voxtral)  # no __init__: we only exercise _consume_tokens
    return v._consume_tokens(iter(stream))


def test_stops_at_first_stop_token_excluded():
    # 2 is a stop token; it and everything after are dropped.
    assert _consume([5, 6, 7, 2, 9, 9, 9]) == [5, 6, 7]


def test_each_stop_token_ends_generation():
    for stop in _Voxtral._STOP_TOKENS:
        assert _consume([8, 9, stop, 8, 8]) == [8, 9], f"stop={stop}"


def test_repetition_loop_passes_through_for_the_degeneracy_detector():
    # No token backstop: a long single-token run is kept in full (up to the stop),
    # so _looks_degenerate can see it and trigger a retry instead of it being
    # truncated to 10 tokens and slipping through.
    assert _consume([3] * 50 + [2]) == [3] * 50


def test_consumes_until_stop_regardless_of_repeats():
    assert _consume([9] * 9 + [1] + [9] * 9 + [2]) == [9] * 9 + [1] + [9] * 9


def test_empty_stream():
    assert _consume([]) == []


def test_immediate_stop_gives_empty():
    assert _consume([2, 5, 6]) == []


def test_stop_tokens_are_control_tokens_never_text():
    """Voxtral's Tekken vocabulary has 131072 entries of which only the first
    1000 are control tokens. mlx_voxtral's default stop set contains 32000 and
    calls it "a potential padding token" -- it is not, it is the ordinary text
    token " Capital", and stopping there truncated every pass that transcribed
    that word (silently: the remaining text reads as clean prose, so neither
    _looks_degenerate nor the loop breaker flags it). <pad> is id 11."""
    stops = set(_Voxtral._STOP_TOKENS)
    assert stops == {2, 4, 11}                      # </s>, [/INST], <pad>
    assert 32000 not in stops
    assert all(t < 1000 for t in stops), "a stop token outside the control range"


def test_stop_tokens_are_read_from_the_build():
    """The class constant is only a fallback; a loaded model uses the ids its
    own processor reports, so a build numbering its controls differently is
    still stopped correctly."""
    from types import SimpleNamespace

    v = _Voxtral.__new__(_Voxtral)
    v.proc = SimpleNamespace(_special_token_ids={"eos": 7, "inst_end": 8,
                                                 "pad": 9, "bos": 1})
    assert v._resolve_stop_tokens() == (7, 8, 9)     # bos is not a stop
    # A processor that reports nothing falls back to the class constant.
    v.proc = SimpleNamespace(_special_token_ids={})
    assert v._resolve_stop_tokens() == _Voxtral._STOP_TOKENS


def test_token_cb_fires_throttled_without_affecting_output(monkeypatch):
    """The liveness heartbeat calls token_cb(collected_count), throttled by wall
    clock, and must not change which tokens are collected."""
    import noScribe.voxtral_engine as v
    # advance monotonic() enough each read that the >=1.5s throttle fires
    ticks = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    monkeypatch.setattr(v.time, "monotonic", lambda: next(ticks, 999.0))
    seen = []
    vox = _Voxtral.__new__(_Voxtral)
    out = vox._consume_tokens(iter([10, 21, 12, 2, 99]), token_cb=seen.append)
    assert out == [10, 21, 12]          # stop token 2 (and 99 after it) dropped
    assert seen and seen[-1] <= 3       # reports the running collected-count


def test_token_cb_absent_is_fine():
    assert _consume([5, 6, 2]) == [5, 6]  # default token_cb=None path unchanged


def test_breaker_gave_up_stops_consumption_early():
    """When the loop breaker gives up mid-generation, _consume_tokens must stop
    pulling tokens immediately instead of running to the stop token."""
    class _GiveUpAfter:
        def __init__(self, n):
            self.n = n
            self.reads = 0
        @property
        def gave_up(self):
            self.reads += 1
            return self.reads > self.n

    v = _Voxtral.__new__(_Voxtral)
    stream = iter([7] * 1000 + [2])
    out = v._consume_tokens(stream, breaker=_GiveUpAfter(5))
    assert len(out) == 5                 # truncated, not the full 1000
    assert next(stream) == 7             # generator not exhausted
