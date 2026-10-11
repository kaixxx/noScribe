"""transcribe() end to end with the model and the aligners stubbed out.

What these guard is plumbing between the pieces rather than any one of them:
what reaches the model as its language, which aligner survives an eviction,
what an empty file or a missing download turns into. Each failed without an
error of its own -- a wrong prompt, memory not coming back, or a message
naming an environment variable instead of the problem.
"""
import gc
import weakref

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from noScribe import voxtral_engine as v  # noqa: E402
from noScribe.voxtral_engine import SAMPLE_RATE  # noqa: E402
from test_loop_breaker import ENGLISH, GERMAN  # noqa: E402


def _wav(tmp_path, seconds):
    p = tmp_path / "t.wav"
    sf.write(p, np.zeros(int(seconds * SAMPLE_RATE), dtype="float32"), SAMPLE_RATE)
    return str(p)


class _Vox:
    """The model: canned texts in call order; records the language asked for."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.languages = []

    def transcribe_array(self, audio, language, max_new_tokens=4096, **kw):
        self.languages.append(language)
        return self.texts[min(len(self.languages) - 1, len(self.texts) - 1)]


class _FakeAligner:
    """An aligner that sits on the GPU, so evicting it asks for the memory back."""
    device = "mps"
    loaded = []

    def __init__(self, model):
        self.model = model
        _FakeAligner.loaded.append(weakref.ref(self))

    def align_words(self, words, audio, t_offset=0.0, depth=0):
        return [{"word": w, "start": t_offset + i * 0.5,
                 "end": t_offset + i * 0.5 + 0.4, "prob": 0.9}
                for i, w in enumerate(words)]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """transcribe() with a fake model, fake aligners and plenty of RAM; the
    repo is a local directory so nothing is looked up on the hub."""
    _FakeAligner.loaded = []
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 64.0)
    monkeypatch.setattr(v, "_Aligner", _FakeAligner)
    monkeypatch.setattr(v, "_unfetchable", lambda model: False)  # no network in tests
    repo = tmp_path / "voxtral-mini-8bit"
    repo.mkdir()

    def run(vox, seconds, **kw):
        monkeypatch.setattr(v, "_Voxtral", lambda r: vox)
        return v.transcribe(_wav(tmp_path, seconds), voxtral_repo=str(repo),
                            chunk_sec=v.HARD_MIN_CHUNK_SEC, **kw)
    return run


# --------------------------------------------------------------------------- #
# The language the caller asked for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelled", ["auto", "Multilingual", ""])
def test_auto_reaches_the_model_as_no_language(stubbed, spelled):
    """"auto" was cut to its first two letters like any code and became the
    pinned language "au": written into the prompt as lang:au, used as the
    target of the translated-pass guard, and it froze the aligner on the
    multilingual model instead of following the text."""
    loaded = []
    orig = v._AlignerPool._load

    def spy(self, model, remember=True):
        loaded.append(model)
        return orig(self, model, remember)

    vox = _Vox(GERMAN, ENGLISH, GERMAN)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v._AlignerPool, "_load", spy)
        stubbed(vox, 130, language=spelled)
    assert vox.languages and set(vox.languages) == {None}, vox.languages
    assert v.ALIGN_MODELS["de"] in loaded and v.ALIGN_MODELS["en"] in loaded, loaded


# --------------------------------------------------------------------------- #
# An evicted aligner really goes
# --------------------------------------------------------------------------- #
def test_an_evicted_aligner_is_not_held_by_the_chunk_loop(stubbed, monkeypatch):
    """transcribe() kept the previous chunk's aligner in a local until the next
    one replaced it -- and the pool evicts before it loads, so at the moment
    of eviction that local was still the last reference. _release_gpu_memory
    then freed nothing (its docstring has the measurement: 2.04 GB stayed
    resident while a local held the model)."""
    alive_at_release = []

    def release():
        gc.collect()
        alive_at_release.append(
            sum(1 for r in _FakeAligner.loaded if r() is not None))

    monkeypatch.setattr(v._AlignerPool, "MAX_CACHED", 1)
    monkeypatch.setattr(v._AlignerPool, "_release_gpu_memory", staticmethod(release))
    stubbed(_Vox(GERMAN, ENGLISH, GERMAN), 130, language=None)
    assert alive_at_release, "no aligner was ever evicted"
    assert alive_at_release == [0] * len(alive_at_release), alive_at_release


# --------------------------------------------------------------------------- #
# An empty file
# --------------------------------------------------------------------------- #
def test_an_empty_file_is_an_empty_transcript(monkeypatch, tmp_path):
    """Zero samples got as far as the log-Mel features, after the model load,
    and died there with "[as_strided] Negative dimensions not allowed".
    Whisper returns no segments for such a file; so does this, without loading
    anything."""
    def no_load(repo):
        raise AssertionError("the model was loaded for an empty file")

    monkeypatch.setattr(v, "_Voxtral", no_load)
    segments, info = v.transcribe(_wav(tmp_path, 0), voxtral_repo="voxtral-mini-8bit")
    assert segments == []
    assert info["duration"] == 0.0


# --------------------------------------------------------------------------- #
# ram_reserve_gb=0
# --------------------------------------------------------------------------- #
def test_a_zero_ram_reserve_is_not_the_default(monkeypatch):
    """0 is falsy, so an explicit "hold nothing back" silently became the 7 GB
    default. The GUI maps its 0 to None before it gets here, so its default is
    unaffected; a direct caller asking for 0 gets 0."""
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 24.0)
    repo = "voxtral-small-4bit"
    default = v._auto_chunk_sec(repo, None, None)
    none_held = v._auto_chunk_sec(repo, None, 0)
    assert none_held > default
    assert none_held == int(v.max_safe_chunk_sec(repo))


# --------------------------------------------------------------------------- #
# A model that is not downloaded, with no internet
# --------------------------------------------------------------------------- #
@pytest.fixture
def hub(monkeypatch):
    """The state of the hub as _load_from_hub sees it: whether the model is in
    the cache, whether the hub answers, whether downloads are switched off."""
    def set_state(missing=True, reachable=False, offline=False):
        monkeypatch.setattr(v, "_needs_download", lambda repo: missing)
        monkeypatch.setattr(v, "_hub_reachable", lambda: reachable)
        monkeypatch.setattr(v, "_hub_offline", lambda: offline)
    return set_state


_OFFLINE = ("Cannot find an appropriate cached snapshot folder for the specified "
            "revision on the local disk and outgoing traffic has been disabled. To "
            "enable repo look-ups and downloads online, set 'HF_HUB_OFFLINE=0' as "
            "environment variable.")


def test_a_missing_voxtral_model_says_so_offline(monkeypatch, tmp_path, hub):
    """The job failed with huggingface_hub's own text, which tells the user to
    set an environment variable. What they need to know is that the model has
    not been downloaded and that it takes one online run."""
    hub_errors = pytest.importorskip("huggingface_hub.utils")

    def offline(repo):
        raise hub_errors.LocalEntryNotFoundError(_OFFLINE)

    hub(missing=True, reachable=False)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 64.0)
    monkeypatch.setattr(v, "_Voxtral", offline)
    repo = v.VOXTRAL_MODELS["voxtral-mini-8bit"]
    with pytest.raises(RuntimeError, match="not downloaded yet") as err:
        v.transcribe(_wav(tmp_path, 5), voxtral_repo=repo, need_timestamps=False)
    assert repo in str(err.value) and "5.6 GB" in str(err.value)
    assert isinstance(err.value.__cause__, hub_errors.LocalEntryNotFoundError)


@pytest.mark.parametrize("error", [
    OSError("Can't load feature extractor for 'x'. If you were trying to load it "
            "from 'https://huggingface.co/models', make sure you don't have a "
            "local directory with the same name."),   # transformers, offline
    TimeoutError("The read operation timed out"),    # a download that stalled
])
def test_a_missing_model_the_hub_cannot_deliver_says_so(monkeypatch, hub, error):
    """Offline, transformers raises a bare OSError for an aligner, with nothing
    chained that says why; a download that stalls ends in an error that is no
    OSError at all. Whether the model could be fetched is asked of the hub."""
    def failing(model):
        raise error

    hub(missing=True, reachable=False)
    monkeypatch.setattr(v, "_Aligner", failing)
    with pytest.raises(RuntimeError, match="alignment model .* not downloaded yet"):
        v._AlignerPool("de", None).aligner_for(GERMAN)


def test_a_load_that_fails_while_the_hub_answers_says_so_itself(monkeypatch, hub):
    """A stale token (401), an unwritable cache or a full disk: the hub is
    there, and "connect once" would be the wrong advice. transformers wraps
    the first two into a bare OSError, and hf_xet reports the last without an
    errno, so none of them can be told apart by the exception."""
    def failing(model):
        raise OSError("There was a specific connection error ... 401 Unauthorized")

    hub(missing=True, reachable=True)
    monkeypatch.setattr(v, "_Aligner", failing)
    with pytest.raises(OSError, match="401") as err:
        v._AlignerPool("de", None).aligner_for(GERMAN)
    assert not isinstance(err.value, RuntimeError)


def test_a_cached_model_that_fails_to_load_says_so_itself(monkeypatch, hub):
    def broken(model):
        raise OSError("corrupt weights")

    hub(missing=False, reachable=False)
    monkeypatch.setattr(v, "_Aligner", broken)
    with pytest.raises(OSError, match="corrupt weights"):
        v._AlignerPool("de", None).aligner_for(GERMAN)


def test_offline_by_setting_says_so(monkeypatch, hub):
    """With HF_HUB_OFFLINE set by the user, connecting would not help."""
    def offline(model):
        raise OSError("Can't load feature extractor")

    hub(missing=True, reachable=True, offline=True)
    monkeypatch.setattr(v, "_Aligner", offline)
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        v._AlignerPool("de", None).aligner_for(GERMAN)


def test_a_chunk_whose_aligner_cannot_be_fetched_uses_the_one_in_use(monkeypatch):
    """Offline on Auto, a chunk that reads as English while only the German
    aligner was ever downloaded ended the whole job after its pass had been
    decoded. The aligner already loaded aligns it instead, with a warning."""
    monkeypatch.setattr(v, "_Aligner", _FakeAligner)
    monkeypatch.setattr(v, "_unfetchable", lambda model: model == v.ALIGN_MODELS["en"])
    logs = []
    pool = v._AlignerPool(None, lambda level, msg: logs.append((level, msg)))
    assert pool.aligner_for(GERMAN).model == v.ALIGN_MODELS["de"]
    assert pool.aligner_for(ENGLISH).model == v.ALIGN_MODELS["de"]
    assert any(level == "warn" and v.ALIGN_MODELS["en"] in msg for level, msg in logs)


def test_a_set_language_whose_aligner_cannot_be_fetched_fails_before_decoding(
        stubbed, monkeypatch):
    """With the language set, its aligner is known before the first pass: the
    job used to decode a whole pass and only then fail on the aligner."""
    monkeypatch.setattr(v, "_unfetchable", lambda model: model == v.ALIGN_MODELS["de"])
    vox = _Vox(GERMAN)
    with pytest.raises(RuntimeError, match="alignment model .* not downloaded yet"):
        stubbed(vox, 70, language="de")
    assert vox.languages == []


def test_a_first_download_is_announced(monkeypatch, hub):
    """A 6 GB download with nothing in the log looks like a hang -- and one
    that cannot happen, offline, is not announced."""
    repo = v.VOXTRAL_MODELS["voxtral-mini-8bit"]
    logs = []

    def load(missing, offline):
        logs.clear()
        hub(missing=missing, reachable=True, offline=offline)
        v._load_from_hub(lambda: "model", "Voxtral model", repo, 5.6,
                         lambda lvl, msg: logs.append(msg))
        return logs

    assert any("Downloading" in m and "5.6 GB" in m for m in load(missing=True, offline=False))
    assert load(missing=False, offline=False) == []
    assert load(missing=True, offline=True) == []


def test_needs_download_asks_the_cache(monkeypatch, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, filename: None)
    assert v._needs_download("some/repo")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, filename: "/cache/config.json")
    assert not v._needs_download("some/repo")
    assert not v._needs_download(str(tmp_path))       # a local directory


def test_a_download_that_stalls_says_so(monkeypatch, hub):
    """When the connection drops during the first download, huggingface_hub
    gives up with httpx's own error -- no OSError, so it slipped past the
    rewording and the user read "The read operation timed out"."""
    httpx = pytest.importorskip("httpx")

    def stalled(model):
        raise httpx.ReadTimeout("The read operation timed out")

    hub(missing=True, reachable=False)
    monkeypatch.setattr(v, "_Aligner", stalled)
    with pytest.raises(RuntimeError, match="not downloaded yet"):
        v._AlignerPool("de", None).aligner_for(GERMAN)


def test_a_refusal_names_what_the_model_really_needs(monkeypatch):
    """The refusal quoted the peak without the headroom it is checked with, so
    an 18 GB Mac read "needs about 17 GB ... does not fit in 18 GB"."""
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 18.0)
    with pytest.raises(MemoryError) as err:
        v.max_safe_chunk_sec("voxtral-small-4bit")
    assert f"about {v.min_ram_gb('voxtral-small-4bit'):.0f} GB" in str(err.value)


def test_a_renamed_model_points_at_one_that_is_offered():
    """A model saved under its old name was silently replaced by the first
    Whisper model; the rename table has to lead to a build that exists."""
    m = pytest.importorskip("noScribe.main")
    for old, new in m.RENAMED_MODELS.items():
        assert new in v.VOXTRAL_MODELS and old not in v.VOXTRAL_MODELS
