import pytest
import utils


def test_str_to_ms():
    tmp = utils.str_to_ms("01:00:00")
    assert tmp == 3600_000

    with pytest.raises(ValueError):
        tmp = utils.str_to_ms("01:00:0X")

    with pytest.raises(TypeError):
        tmp = utils.str_to_ms(12)


def test_get_unique_filename(tmp_path):
    """
    Tests for the `get_unique_filename()` function.
    """

    # Run with an empty directory.
    path_input = tmp_path / "test.txt"
    assert utils.get_unique_filename(str(path_input)) == str(tmp_path / "test.txt")

    # Assume, there are multiple files.
    path_input = tmp_path / "test1.txt"
    path_list = [tmp_path / "test2.txt", tmp_path / "test3.txt"]
    assert utils.get_unique_filename(
        str(path_input), [str(x) for x in path_list]
    ) == str(tmp_path / "test1.txt")

    # Assume, there are multiple files with same name.
    path_input = tmp_path / "test.txt"
    path_list = [tmp_path / "test.txt", tmp_path / "test.txt"]
    assert utils.get_unique_filename(
        str(path_input), [str(x) for x in path_list]
    ) == str(tmp_path / "test_1.txt")

    # Try with file existing in path.
    path_input = tmp_path / "test.txt"
    path_input.touch()
    assert utils.get_unique_filename(
        str(path_input), [str(x) for x in path_list]
    ) == str(tmp_path / "test_1.txt")

    # Try with file existing in path and in file list.
    path_input = tmp_path / "test.txt"
    path_input.touch()
    path_list = [tmp_path / "test.txt", tmp_path / "test.txt"]
    assert utils.get_unique_filename(
        str(path_input), [str(x) for x in path_list]
    ) == str(tmp_path / "test_1.txt")

    # Try with file existing in path and replacement in file list.
    path_input = tmp_path / "test.txt"
    path_input.touch()
    path_list = [tmp_path / "test_1.txt", tmp_path / "test_2.txt"]
    assert utils.get_unique_filename(
        str(path_input), [str(x) for x in path_list]
    ) == str(tmp_path / "test_3.txt")

    # Test a filename without an extension.
    path_input = tmp_path / "test"
    assert utils.get_unique_filename(str(path_input)) == str(tmp_path / "test")
