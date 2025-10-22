import importlib.resources as impres

import pytest
import av

import audio


def test_to_wav_with_expected_input(tmp_path):
    """
    Test the `ToWav` class with expected input.
    """

    # Use whole interview file.
    path_input = impres.files("noScribe") / "tests" / "data" / "interview.mp3"
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
    path_input = impres.files("noScribe") / "tests" / "data" / "interview.mp3"
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

    path_input = impres.files("noScribe") / "tests" / "data" / "interview.mp3"
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
