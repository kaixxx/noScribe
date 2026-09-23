from fractions import Fraction
import importlib.resources as impres
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import av

from noScribe import audio


def test_to_wav_with_expected_input(tmp_path):
    """
    Test the `ToWav` class with expected input.
    """

    # Use whole interview file.
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "interview.wav"

    # Convert
    with audio.convert.ToWav(path_input, path_output) as towav:
        while towav.convert():
            pass

    # Ffmpeg output files are unfortunately not reproducable. Thus, we try to
    # determine if the file got correctly converted by checking the file size.
    # The file is roughly around 22.5MB. Check a lower and upper limit.
    assert path_output.stat().st_size == pytest.approx(22.5 * pow(1024, 2), rel=1e-2)

    # Load the output file and check output with pyav.
    with av.open(path_output) as container:
        stream = container.streams.audio[0]
        assert stream.sample_rate == 16000
        assert stream.format.name == "s16"
        assert stream.channels == 1
        # File is roughly 12m17s long.
        assert stream.duration * stream.time_base == pytest.approx(
            12 * 60 + 17, rel=1e-2
        )


def test_to_wav_overwrites_output_file(tmp_path):
    """
    Test the `ToWav` class that existing files get overwritten only if
    specified.
    """

    # Use whole interview file.
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "interview.wav"

    with audio.convert.ToWav(path_input, path_output) as towav:
        while towav.convert():
            pass

    # Check that another process is only overwriting if `force=True`.
    with pytest.raises(FileExistsError):
        with audio.convert.ToWav(path_input, path_output, force=False) as towav:
            while towav.convert():
                pass

    last_mod = path_output.stat().st_mtime
    with audio.convert.ToWav(path_input, path_output, force=True) as towav:
        while towav.convert():
            pass
    assert last_mod < path_output.stat().st_mtime


def test_to_wav_start_stop_args(tmp_path):
    """
    Test the `ToWav` class that only parts of a file can be converted.
    """

    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "interview-part0.wav"

    # Use a start time for conversion.
    with audio.convert.ToWav(path_input, path_output) as towav:
        # Seek to 6 minutes.
        towav.seek(6 * 60 * 1000)

        while towav.convert():
            pass

    # The file is roughly around 11.5MB. Check a lower and upper limit.
    assert path_output.stat().st_size == pytest.approx(11.5 * pow(1024, 2), rel=1e-2)

    # Load the output file and check output with pyav.
    with av.open(path_output) as container:
        stream = container.streams.audio[0]
        assert stream.sample_rate == 16000
        assert stream.format.container_name == "s16le"
        assert stream.channels == 1
        # File is roughly 6m17s long.
        assert stream.duration * stream.time_base == pytest.approx(
            6 * 60 + 17, rel=1e-2
        )

    # Use a start and end time for conversion.
    path_output = tmp_path / "interview-part1.wav"
    with audio.convert.ToWav(path_input, path_output) as towav:
        # Seek to 6 minutes and end after 7 minutes.
        towav.seek(6 * 60 * 1000)
        towav.stop_after(7 * 60 * 1000)

        while towav.convert():
            pass

    # The file is roughly around 1.85MB. Check a lower and upper limit.
    assert path_output.stat().st_size == pytest.approx(1.85 * pow(1024, 2), rel=1e-2)

    # Load the output file and check output with pyav.
    with av.open(path_output) as container:
        stream = container.streams.audio[0]
        assert stream.sample_rate == 16000
        assert stream.format.container_name == "s16le"
        assert stream.channels == 1
        # File is 1min long.
        assert stream.duration * stream.time_base == pytest.approx(1 * 60, rel=1e-2)


def test_to_wav_skips_invalid_packets(tmp_path, monkeypatch):
    """
    Test that invalid packets do not abort the entire conversion.
    """

    class FakeInvalidDataError(Exception):
        pass

    class FakeFrame:
        def __init__(self, time):
            self.time = time

    class FakePacket:
        def __init__(self, frames=None, exc=None):
            self.frames = list(frames or [])
            self.exc = exc

        def decode(self):
            if self.exc is not None:
                raise self.exc
            return list(self.frames)

    class FakeInputContainer:
        def __init__(self, packets):
            self._packets = packets
            self.streams = SimpleNamespace(audio=["audio-stream"])
            self.closed = False

        def demux(self, stream):
            assert stream == "audio-stream"
            return iter(self._packets)

        def close(self):
            self.closed = True

    class FakeOutputStream:
        def __init__(self):
            self.encoded_frames = []

        def encode(self, frame=None):
            self.encoded_frames.append(frame)
            if frame is None:
                return ["flush-packet"]
            return [f"packet-{frame.time}"]

    class FakeOutputContainer:
        def __init__(self):
            self.stream = FakeOutputStream()
            self.muxed_packets = []
            self.closed = False

        def add_stream(self, codec, rate, layout):
            assert codec == "pcm_s16le"
            assert rate == 16000
            assert layout == "mono"
            return self.stream

        def mux(self, packet):
            self.muxed_packets.append(packet)

        def close(self):
            self.closed = True

    packets = [
        FakePacket(frames=[FakeFrame(0.0)]),
        FakePacket(exc=FakeInvalidDataError("broken mp3 packet")),
        FakePacket(frames=[FakeFrame(0.5)]),
    ]
    input_container = FakeInputContainer(packets)
    output_container = FakeOutputContainer()

    def fake_open(path, mode=None, format=None):
        if mode == "w":
            assert format == "wav"
            return output_container
        return input_container

    fake_av = SimpleNamespace(
        open=fake_open,
        error=SimpleNamespace(InvalidDataError=FakeInvalidDataError),
    )
    monkeypatch.setattr(audio.convert, "av", fake_av)

    path_input = tmp_path / "broken.mp3"
    path_output = tmp_path / "broken.wav"

    with audio.convert.ToWav(path_input, path_output) as towav:
        while towav.convert():
            pass

    assert towav.decode_error_count == 1
    assert len(output_container.stream.encoded_frames) == 3
    assert [frame.time for frame in output_container.stream.encoded_frames[:-1]] == [0.0, 0.5]
    assert output_container.stream.encoded_frames[-1] is None
    assert output_container.muxed_packets == ["packet-0.0", "packet-0.5", "flush-packet"]
    assert input_container.closed is True
    assert output_container.closed is True


def _write_wav_counting_seconds(path, seconds):
    """A 16 kHz WAV whose samples hold the number of the second they are in,
    times 1000, so a converted file shows where it starts."""
    samples = np.repeat(np.arange(seconds, dtype=np.int16) * 1000, 16000)
    sf.write(path, samples, 16000)


def test_to_wav_start_stop_in_wav_input(tmp_path):
    """
    WAV reports no start time for its stream (`start_time is None`), which
    made `seek` fail with "unsupported operand type(s) for *: 'NoneType' and
    'Fraction'" -- a start time could not be used on any WAV file.
    """

    path_input = tmp_path / "counting.wav"
    path_output = tmp_path / "part.wav"
    _write_wav_counting_seconds(path_input, 10)
    with av.open(path_input) as container:
        assert container.streams.audio[0].start_time is None

    with audio.convert.ToWav(path_input, path_output) as towav:
        towav.seek(4 * 1000)
        towav.stop_after(7 * 1000)
        while towav.convert():
            pass

    samples, _ = sf.read(path_output, dtype="int16")
    assert samples[0] == 4000
    assert len(samples) / 16000 == pytest.approx(3, abs=0.1)


@pytest.mark.parametrize(
    "samples, rate, start, stop",
    [
        (10 * 16000, 16000, 12 * 1000, 0),
        (10 * 16000, 16000, 6 * 1000, 5 * 1000),
        # The last 45 samples at 44.1 kHz resample to fewer than the resampler
        # hands on: a frame reaches the encoder, but no packet leaves it.
        (3 * 44100 + 45, 44100, 3 * 1000, 0),
    ],
)
def test_to_wav_says_when_there_is_nothing_to_convert(tmp_path, samples, rate, start, stop):
    """
    A start time at or after the end, or a stop time before the start, leaves
    nothing to convert -- and PyAV creates the output file only with the first
    packet it writes, so the next step failed on a missing file. Now that
    seeking works on WAV, it is easy to get there; the conversion says so
    itself.
    """

    path_input = tmp_path / "input.wav"
    sf.write(path_input, np.zeros(samples, dtype=np.int16), rate)

    with pytest.raises(ValueError, match="No audio to convert"):
        with audio.convert.ToWav(path_input, tmp_path / "part.wav") as towav:
            towav.seek(start)
            if stop:
                towav.stop_after(stop)
            while towav.convert():
                pass
    assert not (tmp_path / "part.wav").exists()


class _SeeksToTheStart:
    """An input container whose index is too coarse to land anywhere but at
    the start, like an audio-only WebM or Matroska file with cues only at its
    clusters, seconds apart."""

    def __init__(self, container):
        self._container = container

    def seek(self, offset, stream):
        self._container.seek(0, stream=stream)

    def close(self):
        self._container.close()


def test_to_wav_drops_what_the_seek_lands_on_before_the_start(tmp_path):
    """
    `seek` lands at or before the start -- at the last index point, which in
    some containers is seconds before it. Everything from there on used to be
    converted, so the transcript began early and every timestamp was late.
    What the seek lands on before the start is dropped, to within one frame.
    """

    path_input = tmp_path / "counting.wav"
    path_output = tmp_path / "part.wav"
    _write_wav_counting_seconds(path_input, 10)
    with av.open(path_input) as container:
        frame_samples = next(container.decode(audio=0)).samples

    with audio.convert.ToWav(path_input, path_output) as towav:
        towav.container_input = _SeeksToTheStart(towav.container_input)
        towav.seek(4 * 1000)
        towav.stop_after(7 * 1000)
        while towav.convert():
            pass

    samples, _ = sf.read(path_output, dtype="int16")
    assert samples[0] >= 3000
    assert np.count_nonzero(samples < 4000) < frame_samples
    assert len(samples) / 16000 == pytest.approx(3, abs=0.2)


def test_to_wav_start_stop_count_from_stream_start(tmp_path):
    """
    Not every stream starts at zero: MP3 skips the encoder delay (25 ms in
    tests/data/interview.mp3), MPEG-TS keeps a running clock. The start
    position used to be counted back from the stream's start time instead of
    forward, so it landed twice that early; frame times include it, so the
    stop position has to add it as well. The time base of MP3 in AVI is
    32/1225 s, which multiplying by its denominator got wrong.
    """

    seeks = []
    towav = audio.convert.ToWav(tmp_path / "in.avi", tmp_path / "out.wav")
    towav.stream_input = SimpleNamespace(time_base=Fraction(32, 1225), start_time=1225)
    towav.container_input = SimpleNamespace(
        seek=lambda offset, stream: seeks.append(offset)
    )

    towav.seek(4 * 1000)
    towav.stop_after(7 * 1000)

    # The stream starts at 1225 * 32/1225 = 32 s.
    assert seeks == [int(36 / Fraction(32, 1225))]
    assert towav.stop_after_sec == pytest.approx(39)
