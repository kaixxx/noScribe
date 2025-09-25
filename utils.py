"""
Different small and distinct helper functions
"""

import i18n


def str_to_ms(time_str: str) -> int:
    """
    Convert "hh:mm:ss" string into milliseconds
    """

    try:
        # See https://stackoverflow.com/a/6402859
        h, m, s = time_str.split(":")
        ret = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    except ValueError as e:
        raise ValueError(
            "InvalidTimeString", i18n.t("err_invalid_time_string"), time_str
        ) from e
    except AttributeError as e:
        raise TypeError(
            "InvalidTimeType",
            i18n.t("err_invalid_time_string"),
            type(time_str),
            time_str,
        ) from e

    return ret
