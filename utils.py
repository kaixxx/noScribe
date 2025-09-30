"""
Different small and distinct helper functions
"""

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


def ms_to_str(milliseconds: int, include_ms: bool = False) -> str:
    """
    Convert milliseconds to a formatted timestamp string in "hh:mm:ss" format.

    Args:
        milliseconds (float): The number of milliseconds to convert.
        include_ms (bool, optional): Whether to include milliseconds in the
        output. Defaults to False.

    Returns:
        str: A formatted timestamp string.
    """

    if milliseconds > 86400000:
        raise ValueError("milliseconds are larger than 24 hours", milliseconds)
    if milliseconds < 0:
        raise ValueError("milliseconds smaller than zero", milliseconds)

    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    if include_ms:
        formatted += f".{milliseconds:03d}"

    return formatted
