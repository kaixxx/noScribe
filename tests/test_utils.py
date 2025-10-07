from pathlib import Path

import pytest
import utils


def test_str_to_ms():
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
