import importlib.resources as impres
from types import SimpleNamespace

import pytest
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
