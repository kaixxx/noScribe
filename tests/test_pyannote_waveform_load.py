"""Prove that loading the diarization waveform via soundfile is a drop-in
replacement for the previous ``torchaudio.load`` call.

The pyannote worker only ever loads noScribe's own converted audio
(16 kHz mono pcm_s16le WAV, see noScribe/audio/convert.py) and passes it to
the pipeline as an in-memory ``{"waveform": tensor, "sample_rate": int}``
dict.  So the loader just has to produce the exact same tensor -- which this
test asserts bit-for-bit against torchaudio while both libraries are
installed side by side.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
# Importing the worker triggers the package __init__, which pulls in the GUI --
# on a Python built without tk, that would fail collection instead of skipping.
pytest.importorskip("tkinter")

from noScribe.pyannote_mp_worker import load_waveform


@pytest.fixture()
def converted_wav(tmp_path):
    """A WAV exactly like noScribe's conversion step writes: 16 kHz mono PCM16."""
    path = tmp_path / "converted.wav"
    signal = np.random.default_rng(0).uniform(-1.0, 1.0, 16000 * 3)
    sf.write(path, signal, 16000, subtype="PCM_16")
    return path


def test_load_waveform_shape_dtype_rate(converted_wav):
    waveform, sample_rate = load_waveform(str(converted_wav))
    assert sample_rate == 16000
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 2 and waveform.shape[0] == 1  # (channels, frames)
    assert waveform.shape[1] == 16000 * 3
    assert waveform.is_contiguous()


def test_load_waveform_rejects_unconverted_input_with_context(tmp_path):
    # A file that never went through noScribe's conversion step must fail
    # with an error that points at the conversion contract, not a bare
    # libsndfile message.
    bogus = tmp_path / "not_converted.m4a"
    bogus.write_bytes(b"\x00\x00\x00\x20ftypM4A this is not a wav")
    with pytest.raises(RuntimeError, match="audio/convert.py"):
        load_waveform(str(bogus))


def test_load_waveform_reports_a_missing_file_as_missing(tmp_path):
    # soundfile raises LibsndfileError for a missing path just as it does for a
    # corrupt one, so without the explicit check a WAV that vanished would be
    # blamed on the conversion step.
    with pytest.raises(FileNotFoundError):
        load_waveform(str(tmp_path / "never_written.wav"))


def test_load_waveform_bit_identical_to_torchaudio(converted_wav):
    # Migration-time proof: runs only while torchaudio is still installed and
    # may be deleted once torchaudio leaves the tested stacks. The test above
    # keeps covering the loader on its own.
    torchaudio = pytest.importorskip("torchaudio")
    expected, expected_rate = torchaudio.load(str(converted_wav))
    actual, actual_rate = load_waveform(str(converted_wav))
    assert actual_rate == expected_rate
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)  # bit-for-bit, not just allclose
