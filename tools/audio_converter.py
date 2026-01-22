#!/usr/bin/env python3
"""
Audio Converter - WAV to MP3 conversion for Gemini audio processing.

Extracts the microphone channel (channel 3) from multi-channel recordings
and converts to high-quality MP3 for efficient upload to Gemini API.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Absolute paths for launchd compatibility
FFMPEG_PATH = '/opt/homebrew/bin/ffmpeg'
FFPROBE_PATH = '/opt/homebrew/bin/ffprobe'


@dataclass
class AudioInfo:
    """Information about an audio file."""
    channels: int
    sample_rate: int
    duration_seconds: float
    file_size_bytes: int


def get_audio_info(audio_path: Path) -> AudioInfo:
    """Get detailed information about an audio file using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        AudioInfo with channels, sample rate, duration, and file size

    Raises:
        RuntimeError: If ffprobe fails
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels,sample_rate:format=duration',
        '-of', 'csv=p=0:s=,',
        str(audio_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    # Parse output: "sample_rate,channels\nduration"
    lines = result.stdout.strip().split('\n')
    stream_info = lines[0].split(',')

    sample_rate = int(stream_info[0])
    channels = int(stream_info[1])
    duration = float(lines[1]) if len(lines) > 1 else 0.0

    return AudioInfo(
        channels=channels,
        sample_rate=sample_rate,
        duration_seconds=duration,
        file_size_bytes=audio_path.stat().st_size
    )


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def get_channel_count(audio_path: Path) -> int:
    """Get the number of audio channels.

    Args:
        audio_path: Path to the audio file

    Returns:
        Number of channels
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels',
        '-of', 'csv=p=0',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return int(result.stdout.strip())


def convert_to_mp3(
    input_path: Path,
    output_path: Optional[Path] = None,
    extract_channel: Optional[int] = None,
    quality: int = 2
) -> Path:
    """Convert audio file to high-quality MP3 optimized for speech recognition.

    For MeetingRecorder 3-channel audio:
    - Channel 0-1: System audio (stereo)
    - Channel 2: Microphone (mono) <- This is what we want

    Args:
        input_path: Path to input audio file (WAV, etc.)
        output_path: Path for output MP3 (default: same name with .mp3)
        extract_channel: Channel index to extract (0-based).
                        If None, auto-detects: extracts channel 2 for 3-channel,
                        or mixes all channels for stereo.
        quality: LAME quality setting (0=best, 9=worst). Default 2 (~190kbps VBR)

    Returns:
        Path to the converted MP3 file

    Raises:
        RuntimeError: If ffmpeg conversion fails
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_path = Path(output_path)

    # Get channel count to determine extraction strategy
    num_channels = get_channel_count(input_path)
    logger.info(f"Input file has {num_channels} channels")

    # Determine channel extraction
    if extract_channel is not None:
        # User specified a channel
        channel_idx = extract_channel
    elif num_channels == 3:
        # MeetingRecorder format: extract mic channel (index 2)
        channel_idx = 2
        logger.info("Detected 3-channel audio, extracting channel 3 (mic)")
    elif num_channels == 1:
        # Already mono, no extraction needed
        channel_idx = None
        logger.info("Input is mono, no channel extraction needed")
    else:
        # Stereo or other - mix to mono
        channel_idx = None
        logger.info(f"Input has {num_channels} channels, will mix to mono")

    # Build ffmpeg command
    cmd = [FFMPEG_PATH, '-y', '-i', str(input_path)]

    if channel_idx is not None:
        # Extract specific channel and duplicate to stereo for compatibility
        cmd.extend(['-af', f'pan=stereo|c0=c{channel_idx}|c1=c{channel_idx}'])
    else:
        # Mix to mono then duplicate to stereo
        cmd.extend(['-ac', '1'])

    # High-quality MP3 encoding
    cmd.extend([
        '-codec:a', 'libmp3lame',
        '-qscale:a', str(quality),  # VBR quality (2 = ~190kbps)
        str(output_path)
    ])

    logger.debug(f"FFmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    output_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Converted {input_path.name} -> {output_path.name} ({output_size_mb:.1f} MB)")

    return output_path


def convert_for_gemini(
    input_path: Path,
    output_dir: Optional[Path] = None
) -> Path:
    """Convert audio file optimized for Gemini API upload.

    This is the main entry point for the Gemini audio pipeline.
    It extracts the mic channel from MeetingRecorder recordings
    and converts to high-quality MP3.

    Args:
        input_path: Path to input WAV file
        output_dir: Directory for output MP3 (default: same directory as input)

    Returns:
        Path to the converted MP3 file
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.mp3"

    return convert_to_mp3(
        input_path=input_path,
        output_path=output_path,
        extract_channel=None,  # Auto-detect
        quality=2  # High quality for clear transcription
    )


if __name__ == "__main__":
    # Simple test
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python audio_converter.py <input.wav> [output.mp3]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Show audio info
    info = get_audio_info(input_file)
    print(f"Audio info: {info.channels} channels, {info.sample_rate}Hz, {info.duration_seconds:.1f}s")

    # Convert
    result = convert_to_mp3(input_file, output_file)
    print(f"Output: {result}")
