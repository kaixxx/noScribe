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
        self.stop_after_sec: float = None
        self.decode_error_count: int = 0
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
        self._output_flushed = False

        return self

    def close(self):
        """
        Close the file descriptors for the audio conversion.
        """

        if not self._output_flushed and self.container_output is not None:
            try:
                for packet in self.stream_output.encode(None):
                    self.container_output.mux(packet)
            except Exception as exc:
                logger.warning(
                    "Failed to flush converted audio stream for %s: %s",
                    self.file_output,
                    exc,
                )
            finally:
                self._output_flushed = True

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
        Seeks in the stream approximately to the given milliseconds. This way,
        conversion starts at this point.

        Needs to be called after `open` was called.
        """

        seconds = milliseconds / 1000.0

        # See https://github.com/PyAV-Org/PyAV/blob/main/tests/test_seek.py for
        # more examples on the approach.
        # See also this documentation:
        # https://pyav.org/docs/develop/api/audio.html#module-av.audio.stream
        #
        # `seek` jumps in the stream based on time base (fractions of a
        # second). Thus, using the denominator to get the seek position by
        # multiplying with seconds.

        # Get time base.
        time_base = self.stream_input.time_base

        # Take start time into consideration.
        start_time = self.stream_input.start_time * time_base

        # Seek.
        seek_to = (seconds - start_time) * time_base.denominator
        self.container_input.seek(int(seek_to), stream=self.stream_input)

    def stop_after(self, milliseconds: int):
        """
        Define after how many milliseconds conversion should stop.

        Be careful as this function will not check whether milliseconds are
        greater than the current position of the stream. Thus, if milliseconds
        is greater than the current position, the output file will be empty.
        """

        self.stop_after_sec = milliseconds / 1000.0

    def convert(self) -> bool:
        """
        Convert a frame from the input file to wave output.
        """

        while not self.pending_frames:
            try:
                packet = next(self.packet_iterator)
            except StopIteration:
                return False

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

        # Check whether we are already past the stop time.
        if self.stop_after_sec and frame.time is not None and self.stop_after_sec < frame.time:
            return False

        # Otherwise convert frame.
        for packet in self.stream_output.encode(frame):
            self.container_output.mux(packet)

        return True
