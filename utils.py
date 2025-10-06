"""
Different small and distinct helper functions
"""

import os

from pathlib import Path

import i18n


def str_to_ms(time_str: str) -> int:
    """
    Convert a "hh:mm:ss" time string to milliseconds.

    Args:
        time_str (str): The time string in "hh:mm:ss" format.

    Returns:
        int: The time in milliseconds.

    Raises:
        ValueError: If the input time string is invalid.
        TypeError: If the input is not a string.
    """

    try:
        # See https://stackoverflow.com/a/6402859
        h, m, s = time_str.split(":")
        ret = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    except ValueError as e:
        raise ValueError(
            "time string is invalid", i18n.t("err_invalid_time_string"), time_str
        ) from e
    except AttributeError as e:
        raise TypeError(
            "time string is not of type str",
            i18n.t("err_invalid_time_string"),
            type(time_str),
            time_str,
        ) from e

    return ret


def get_unique_filename(fn: str, file_list=[]) -> str:
    if os.path.exists(fn) or fn in file_list:
        i = 1
        path = Path(fn)
        base_path = os.path.join(path.parent, path.stem)
        file_ext = os.path.splitext(fn)[1][1:] 
        while os.path.exists(f'{base_path}_{i}.{file_ext}') or f'{base_path}_{i}.{file_ext}' in file_list:
            i += 1
            if i > 999:
                break
        return f'{base_path}_{i}.{file_ext}'
    else:
        return fn
