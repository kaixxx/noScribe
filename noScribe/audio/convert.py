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

    output_format = "wav"
    output_codec = "pcm_s16le"
    output_rate = 16000
    output_layout = "mono"
    output_bit_rate = None

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
        # Media position of the first converted sample, for transcript timestamps.
        self.start_offset_ms: int = 0
        self._timeline_start_sec: float = None
        self._first_frame = True
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
        self.container_output = av.open(
            self.file_output, mode="w", format=self.output_format
        )
        self.stream_input = self.container_input.streams.audio[0]
        self.stream_output = self.container_output.add_stream(
            self.output_codec,
            rate=self.output_rate,
            layout=self.output_layout,
        )
        if self.output_bit_rate is not None:
            self.stream_output.bit_rate = self.output_bit_rate
        self.packet_iterator = self.container_input.demux(self.stream_input)
        self.pending_frames.clear()
        self.decode_error_count = 0
        self._packets_written = 0
        self.start_offset_ms = 0
        self._first_frame = True
        self._output_flushed = False
        self._timeline_start_sec = None
        self._start_time()

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
        of the media timeline (the playback timeline for video).

        Needs to be called after `open` was called.
        """

        # Frame times and seek targets include the media timeline's origin.
        self.start_at_sec = milliseconds / 1000.0 + self._start_time()
        self._validate_range()

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
        of the media timeline (like `seek`, not counted from the seek position). A
        position before the seek position leaves nothing to convert.

        Needs to be called after `open` was called.
        """

        # Frame times include the media timeline's origin, see `seek`.
        self.stop_after_sec = milliseconds / 1000.0 + self._start_time()
        self._validate_range()

    def _validate_range(self):
        if (
            self.stop_after_sec is not None
            and self.stop_after_sec <= (
                self.start_at_sec if self.start_at_sec is not None else self._start_time()
            )
        ):
            raise ValueError("No audio to convert: the stop time must be after the start time.")

    def _start_time(self) -> float:
        """
        Origin of the media timeline in seconds. For video, the audio track
        may start after the picture; use the container's origin so start/stop
        positions agree with playback. Audio-only files count from the audio
        stream's first sample. Not every container starts at zero (MP3 skips
        the encoder delay, MPEG-TS keeps a running
        clock), and some, WAV among them, report no start time at all.
        """

        if self._timeline_start_sec is None:
            if self.container_input.streams.video and self.container_input.start_time is not None:
                self._timeline_start_sec = self.container_input.start_time / av.time_base
            elif self.stream_input.start_time is None:
                self._timeline_start_sec = 0.0
            else:
                self._timeline_start_sec = float(self.stream_input.start_time * self.stream_input.time_base)
        return self._timeline_start_sec

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
        if self._first_frame:
            # A backward seek can leave part of a frame before the requested
            # start, or a delayed video audio track can begin after it. Keep
            # transcript timestamps anchored to the samples actually written.
            if frame.time is not None:
                self.start_offset_ms = round((frame.time - self._start_time()) * 1000)
            elif self.start_at_sec is not None:
                self.start_offset_ms = round((self.start_at_sec - self._start_time()) * 1000)
            self._first_frame = False
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


class ToFlac(ToWav):
    """Convert an arbitrary input to lossless mono FLAC for network transfer."""

    output_format = "flac"
    output_codec = "flac"
    output_rate = 16000
    output_layout = "mono"
    output_bit_rate = None
