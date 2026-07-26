"""Prove that loading the diarization waveform via soundfile is a drop-in
replacement for the previous ``torchaudio.load`` call.

The pyannote worker only ever loads the WAV that ``noScribe.audio.convert``
writes and hands it to the pipeline as an in-memory
``{"waveform": tensor, "sample_rate": int}`` dict, so the loader has to produce
the exact same tensor. The fixture goes through the real conversion step rather
than writing a WAV with soundfile itself: the production input is muxed by
PyAV and carries a different header, and a file written and read by the same
library would hide any quirk specific to the other one.
"""
import importlib.resources as impres

import numpy as np
import pytest
import soundfile as sf
import torch

from noScribe import audio
from noScribe.pyannote_mp_worker import load_waveform


@pytest.fixture()
def converted_wav(tmp_path):
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "converted.wav"
    with audio.convert.ToWav(path_input, path_output) as towav:
        towav.stop_after(3000)          # 3 s is plenty and keeps the test quick
        while towav.convert():
            pass
    return path_output


def test_load_waveform_shape_dtype_rate(converted_wav):
    waveform, sample_rate = load_waveform(str(converted_wav))
    assert sample_rate == 16000
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 2 and waveform.shape[0] == 1  # (channels, frames)
    assert waveform.shape[1] > 0
    assert waveform.is_contiguous()


def test_multichannel_is_returned_contiguous(tmp_path):
    """Mono is contiguous either way, so only a stereo input can show that the
    transpose is actually made contiguous before pyannote sees it."""
    path = tmp_path / "stereo.wav"
    sig = np.random.default_rng(0).uniform(-1.0, 1.0, (16000, 2))
    sf.write(path, sig, 16000, subtype="PCM_16")
    waveform, _ = load_waveform(str(path))
    assert waveform.shape == (2, 16000)
    assert waveform.is_contiguous()


def test_undecodable_input_names_the_format(tmp_path):
    # Undecodable, not unconverted: libsndfile reads MP3, FLAC and Ogg quite
    # happily, so the loader cannot tell whether a file went through the
    # conversion step -- only whether it can read it at all.
    bogus = tmp_path / "not_audio.bin"
    bogus.write_bytes(b"\x00\x00\x00\x20ftypM4A this is not a wav")
    with pytest.raises(RuntimeError, match="not a WAV"):
        load_waveform(str(bogus))


def test_unreadable_file_is_not_blamed_on_the_format(tmp_path):
    """libsndfile reports a locked file through the same exception as a bad
    format; only the code tells them apart. Confusing the two sends the user
    to debug the conversion step over a permissions problem."""
    path = tmp_path / "locked.wav"
    sf.write(path, np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    path.chmod(0o000)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            load_waveform(str(path))
    finally:
        path.chmod(0o600)
    assert "not a WAV" not in str(excinfo.value)


def test_empty_audio_is_reported_as_empty(tmp_path):
    """A WAV with a header but no frames would otherwise reach pyannote as a
    (1, 0) tensor and come back as a complaint about tensor layout."""
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype="float32"), 16000, subtype="PCM_16")
    with pytest.raises(RuntimeError, match="no audio"):
        load_waveform(str(path))


def test_load_waveform_bit_identical_to_torchaudio(converted_wav):
    # Migration-time proof: runs only while torchaudio is still installed and
    # may be deleted once torchaudio leaves the tested stacks. The tests above
    # keep covering the loader on its own.
    torchaudio = pytest.importorskip("torchaudio")
    try:
        expected, expected_rate = torchaudio.load(str(converted_wav))
    except ImportError as e:
        # torchaudio >= 2.9 decodes through torchcodec, which needs system
        # FFmpeg libraries noScribe deliberately does not depend on. A missing
        # comparison baseline is not a failure of the loader.
        pytest.skip(f"torchaudio cannot decode here: {e}")
    actual, actual_rate = load_waveform(str(converted_wav))
    assert actual_rate == expected_rate
    assert torch.equal(actual, expected)  # bit-for-bit, not just allclose
