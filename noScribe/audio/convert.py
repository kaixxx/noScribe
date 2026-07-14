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

    def __init__(
        self,
        file_input: Path,
        file_output: Path,
        force: bool = False,
        speed: float = 1.0,
    ):
        # Check whether output path exists. Only overwrite if `force=True`.
        if file_output.exists() and not force:
            raise FileExistsError(file_output)
        if speed <= 0:
            raise ValueError("speed must be greater than zero")

        self.file_input: Path = file_input
        self.file_output: Path = file_output
        self.speed: float = speed
        self.container_input: av.container.Container = None
        self.container_output: av.container.Container = None
        self.stream_input: av.stream.Stream = None
        self.stream_output: av.stream.Stream = None
        self.packet_iterator = None
        self.pending_frames = deque()
        self.stop_after_sec: float = None
        self.decode_error_count: int = 0
        self.filter_graph = None
        self._filter_graph_flushed = False
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
        self.filter_graph = None
        if self.speed != 1.0:
            self.filter_graph = av.filter.Graph()
            last_node = self.filter_graph.add_abuffer(template=self.stream_input)
            for tempo_arg in self._build_atempo_filter_args():
                next_node = self.filter_graph.add("atempo", tempo_arg)
                last_node.link_to(next_node)
                last_node = next_node
            last_node.link_to(self.filter_graph.add("abuffersink"))
            self.filter_graph.configure()
        self.packet_iterator = self.container_input.demux(self.stream_input)
        self.pending_frames.clear()
        self.decode_error_count = 0
        self._filter_graph_flushed = False
        self._output_flushed = False

        return self

    def close(self):
        """
        Close the file descriptors for the audio conversion.
        """

        if self.filter_graph is not None and not self._filter_graph_flushed:
            try:
                self.filter_graph.push(None)
                self._encode_filtered_frames()
            except Exception as exc:
                logger.warning(
                    "Failed to flush audio speed filter for %s: %s",
                    self.file_output,
                    exc,
                )
            finally:
                self._filter_graph_flushed = True

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

    def _build_atempo_filter_args(self) -> list[str]:
        remaining_speed = self.speed
        tempo_args: list[str] = []

        while remaining_speed > 2.0:
            tempo_args.append("2.0")
            remaining_speed /= 2.0

        while remaining_speed < 0.5:
            tempo_args.append("0.5")
            remaining_speed /= 0.5

        tempo_args.append(f"{remaining_speed:.10g}")
        return tempo_args

    @staticmethod
    def _is_filter_graph_empty(exc: Exception) -> bool:
        return type(exc).__name__ in {"BlockingIOError", "EOFError"}

    def _encode_filtered_frames(self):
        while True:
            try:
                frame = self.filter_graph.pull()
            except Exception as exc:
                if self._is_filter_graph_empty(exc):
                    return
                raise

            for packet in self.stream_output.encode(frame):
                self.container_output.mux(packet)

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
        if self.filter_graph is not None:
            self.filter_graph.push(frame)
            self._encode_filtered_frames()
        else:
            for packet in self.stream_output.encode(frame):
                self.container_output.mux(packet)

        return True
