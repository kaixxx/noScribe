from pathlib import Path

import pytest
import utils


def test_str_to_ms():
    """
    Tests for the `str_to_ms` function.
    """

    tmp = utils.str_to_ms("01:00:00")
    assert tmp == 3600_000

    with pytest.raises(ValueError):
        tmp = utils.str_to_ms("01:00:0X")

    with pytest.raises(TypeError):
        tmp = utils.str_to_ms(12)


def test_create_unique_filenames(tmp_path):
    """
    Tests for the `create_unique_filenames()` function.
    """

    # Run with an empty directory.
    path_list = [tmp_path / "test.txt"]
    assert utils.create_unique_filenames(path_list) == path_list

    # Assume, there are multiple files.
    path_list = [tmp_path / "test1.txt", tmp_path / "test2.txt", tmp_path / "test3.txt"]
    assert utils.create_unique_filenames(path_list) == path_list

    # Assume, there are multiple files with same name.
    path_list = [tmp_path / "test.txt", tmp_path / "test.txt", tmp_path / "test.txt"]
    assert utils.create_unique_filenames(path_list) == [
        tmp_path / "test.txt",
        tmp_path / "test_1.txt",
        tmp_path / "test_2.txt",
    ]

    # Assume, there are files with the replacement filename.
    path_list = [tmp_path / "test.txt", tmp_path / "test.txt", tmp_path / "test_1.txt"]
    assert utils.create_unique_filenames(path_list) == [
        tmp_path / "test.txt",
        tmp_path / "test_1.txt",
        tmp_path / "test_1_1.txt",
    ]

    # Now, create one file.
    (tmp_path / "test.txt").touch()

    # Try with file existing in path.
    path_list = [tmp_path / "test.txt"]
    assert utils.create_unique_filenames(path_list) == [tmp_path / "test_1.txt"]

    # Try with file existing in path and in file list.
    path_list = [tmp_path / "test.txt", tmp_path / "test.txt", tmp_path / "test.txt"]
    assert utils.create_unique_filenames(path_list) == [
        tmp_path / "test_1.txt",
        tmp_path / "test_2.txt",
        tmp_path / "test_3.txt",
    ]

    # Try with file existing in path and replacement in file list.
    path_list = [tmp_path / "test.txt", tmp_path / "test_1.txt"]
    assert utils.create_unique_filenames(path_list) == [
        tmp_path / "test_1.txt",
        tmp_path / "test_1_1.txt",
    ]

    # Test a filename without an extension.
    path_list = [tmp_path / "test"]
    assert utils.create_unique_filenames(path_list) == path_list

    # Try without a path component
    path_list = [Path("test1.txt"), Path("test2.txt"), Path("test3.txt")]
    assert utils.create_unique_filenames(path_list) == path_list

    # Try with a local path component.
    path_list = [Path("./test1.txt"), Path("./test2.txt"), Path("./test3.txt")]
    assert (
        utils.create_unique_filenames(path_list)
        == path_list
    )

    # Try with a local path component and files in list.
    path_list = [Path("./test.txt"), Path("./test.txt"), Path("./test.txt")]
    assert (
        utils.create_unique_filenames(path_list)
        == [Path("./test.txt"), Path("./test_1.txt"), Path("./test_2.txt")]
    )

    # Raise an error if unique filename could not be found.
    path_list = [Path("f") for _ in range(1000)]
    with pytest.raises(RuntimeError):
        utils.create_unique_filenames(path_list)
  
  
def test_ms_to_str():
    """
    Tests for the `ms_to_str` function.
    """

    # Without ms
    assert utils.ms_to_str(0) == "00:00:00"
    assert utils.ms_to_str(1000) == "00:00:01"
    assert utils.ms_to_str(60000) == "00:01:00"
    assert utils.ms_to_str(3600000) == "01:00:00"

    # With ms
    assert utils.ms_to_str(0, include_ms=True) == "00:00:00.000"
    assert utils.ms_to_str(1000, include_ms=True) == "00:00:01.000"
    assert utils.ms_to_str(1250, include_ms=True) == "00:00:01.250"
    assert utils.ms_to_str(60000, include_ms=True) == "00:01:00.000"
    assert utils.ms_to_str(60250, include_ms=True) == "00:01:00.250"
    assert utils.ms_to_str(3600000, include_ms=True) == "01:00:00.000"
    assert utils.ms_to_str(3600250, include_ms=True) == "01:00:00.250"

    # Test invalid inputs
    with pytest.raises(ValueError):
        utils.ms_to_str(86400001)

    with pytest.raises(TypeError):
        utils.ms_to_str("abc")

    with pytest.raises(ValueError):
        utils.ms_to_str(123.45)

    with pytest.raises(ValueError):
        utils.ms_to_str(-1000)

    with pytest.raises(ValueError):
        utils.ms_to_str(-1000.5)

        
def test_ms_to_webvtt():
    """
    Tests for the `ms_to_webvtt` function.
    """

    assert utils.ms_to_webvtt(0) == "00:00:00.000"
    assert utils.ms_to_webvtt(1000) == "00:00:01.000"
    assert utils.ms_to_webvtt(1250) == "00:00:01.250"
    assert utils.ms_to_webvtt(60000) == "00:01:00.000"
    assert utils.ms_to_webvtt(60250) == "00:01:00.250"
    assert utils.ms_to_webvtt(3600000) == "01:00:00.000"
    assert utils.ms_to_webvtt(3600250) == "01:00:00.250"

    # Test invalid inputs
    with pytest.raises(ValueError):
        utils.ms_to_webvtt(86400001)

    with pytest.raises(TypeError):
        utils.ms_to_webvtt("abc")

    with pytest.raises(ValueError):
        utils.ms_to_webvtt(123.45)

    with pytest.raises(ValueError):
        utils.ms_to_webvtt(-1000)

    with pytest.raises(ValueError):
        utils.ms_to_webvtt(-1000.5)
