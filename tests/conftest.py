from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture
def delayed_audio_video(tmp_path):
    """A video starting at zero whose eight-second audio track starts at 2 s."""
    path = tmp_path / "delayed.mp4"
    samples = np.repeat(np.arange(8, dtype=np.int16) * 1000, 48000)
    with av.open(path, "w") as container:
        audio = container.add_stream("aac", rate=48000, layout="mono")
        video = container.add_stream("mpeg4", rate=10)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        packets = []
        for offset in range(0, len(samples), 1024):
            frame = av.AudioFrame.from_ndarray(
                samples[offset:offset + 1024][None, :], format="s16", layout="mono"
            )
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = offset + 2 * 48000
            packets.extend(audio.encode(frame))
        packets.extend(audio.encode(None))
        for index in range(100):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            packets.extend(video.encode(frame))
        packets.extend(video.encode(None))
        packets.sort(key=lambda packet: packet.dts * packet.time_base)
        for packet in packets:
            container.mux(packet)
    return path
