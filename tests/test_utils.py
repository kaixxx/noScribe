import pytest
import utils


def test_str_to_ms():
    tmp = utils.str_to_ms("01:00:00")
    assert tmp == 3600_000

    with pytest.raises(ValueError):
        tmp = utils.str_to_ms("01:00:0X")

    with pytest.raises(TypeError):
        tmp = utils.str_to_ms(12)
