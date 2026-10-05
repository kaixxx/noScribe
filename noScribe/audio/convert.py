"""
All classes and functions related to audio conversion.
"""

from collections import deque
from pathlib import Path
import logging

import av

logger = logging.getLogger(__name__)


class ToWav:
    """
    Convert an arbitrary file to wave format.
    """

    def __init__(self, file_input: Path, file_output: Path, force: bool = False):
        # Check whether output path exists. Only overwrite if `force=True`.
        if file_output.exists() and not force:
            raise FileExistsError(file_output)

        self.file_input: Path = file_input
        self.file_output: Path = file_output
        self.container_input: av.container.Container = None
        self.container_output: av.container.Container = None
        self.stream_input: av.stream.Stream = None
        self.stream_output: av.stream.Stream = None
        self.packet_iterator = None
        self.pending_frames = deque()
        self.start_at_sec: float = None
        self.stop_after_sec: float = None
        self.decode_error_count: int = 0
        self._packets_written: int = 0
        self._output_flushed = False

    def open(self):
        """
        Prepares everything to run the audio conversion. The caller must make
        sure to call the `close()` command as well.
        """

        logger.debug(
            "Starting audio conversion to wav: %s -> %s", self.file_input, self.file_output
        )

        self.container_input = av.open(self.file_input)
        self.container_output = av.open(self.file_output, mode="w", format="wav")
        self.stream_input = self.container_input.streams.audio[0]
        self.stream_output = self.container_output.add_stream(
            "pcm_s16le", rate=16000, layout="mono"
        )
        self.packet_iterator = self.container_input.demux(self.stream_input)
        self.pending_frames.clear()
        self.decode_error_count = 0
        self._packets_written = 0
        self._output_flushed = False

        return self

    def close(self):
        """
        Close the file descriptors for the audio conversion.
        """

        self._flush()

        if self.container_input is not None:
            self.container_input.close()
            self.container_input = None

        if self.container_output is not None:
            self.container_output.close()
            self.container_output = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        return False

    def seek(self, milliseconds: int):
        """
        Start the conversion at this position, in milliseconds from the start
        of the stream.

        Needs to be called after `open` was called.
        """

        # Positions count from the start of the stream; frame times and seek
        # targets include its start time.
        self.start_at_sec = milliseconds / 1000.0 + self._start_time()

        # See https://github.com/PyAV-Org/PyAV/blob/main/tests/test_seek.py for
        # more examples on the approach.
        # See also this documentation:
        # https://pyav.org/docs/develop/api/audio.html#module-av.audio.stream
        #
        # `seek` jumps in the stream based on time base (fractions of a
        # second). Thus, dividing the seconds by the time base gives the seek
        # position. It lands at or before it -- in some containers seconds
        # before, at the last index point -- so `convert` drops the frames
        # that end before the start.
        seek_to = self.start_at_sec / self.stream_input.time_base
        self.container_input.seek(int(seek_to), stream=self.stream_input)

    def stop_after(self, milliseconds: int):
        """
        Stop the conversion at this position, in milliseconds from the start
        of the stream (like `seek`, not counted from the seek position). A
        position before the seek position leaves nothing to convert.

        Needs to be called after `open` was called.
        """

        # Frame times include the stream's start time, see `seek`.
        self.stop_after_sec = milliseconds / 1000.0 + self._start_time()

    def _start_time(self) -> float:
        """
        Time of the stream's first sample in seconds. Not every container
        starts at zero (MP3 skips the encoder delay, MPEG-TS keeps a running
        clock), and some, WAV among them, report no start time at all.
        """

        if self.stream_input.start_time is None:
            return 0.0
        return float(self.stream_input.start_time * self.stream_input.time_base)

    def convert(self) -> bool:
        """
        Convert a frame from the input file to wave output.
        """

        while not self.pending_frames:
            try:
                packet = next(self.packet_iterator)
            except StopIteration:
                return self._finished()

            try:
                self.pending_frames.extend(packet.decode())
            except av.error.InvalidDataError as exc:
                self.decode_error_count += 1
                logger.warning(
                    "Skipping invalid audio packet %s while decoding %s: %s",
                    self.decode_error_count,
                    self.file_input,
                    exc,
                )

        frame = self.pending_frames.popleft()

        # Drop what the seek landed on before the start.
        if (
            self.start_at_sec is not None
            and frame.time is not None
            and frame.time + frame.samples / frame.sample_rate <= self.start_at_sec
        ):
            return True

        # Check whether we are already past the stop time.
        if self.stop_after_sec is not None and frame.time is not None and self.stop_after_sec < frame.time:
            return self._finished()

        # Otherwise convert frame.
        for packet in self.stream_output.encode(frame):
            self.container_output.mux(packet)
            self._packets_written += 1

        return True

    def _flush(self):
        """
        Write out what the encoder still holds.
        """

        if self._output_flushed or self.container_output is None:
            return
        try:
            for packet in self.stream_output.encode(None):
                self.container_output.mux(packet)
                self._packets_written += 1
        except Exception as exc:
            logger.warning(
                "Failed to flush converted audio stream for %s: %s",
                self.file_output,
                exc,
            )
        finally:
            self._output_flushed = True

    def _finished(self) -> bool:
        """
        End the conversion. PyAV creates the output file with the first packet
        it writes; without one there is no file at all, and the next step would
        fail on its absence instead.
        """

        self._flush()
        if not self._packets_written:
            raise ValueError(
                "No audio to convert: the file holds none that can be decoded, "
                "or the start time lies at or after its end or after the stop time."
            )
        return False
