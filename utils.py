"""
Different small and distinct helper functions
"""

import html
import html.parser
from pathlib import Path

import i18n

HTML_BLOCK_LEVEL_ELEMENTS = {
    "p",
    "div",
    "ul",
    "ol",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


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


def create_unique_filenames(path_inputs: [Path]) -> Path:
    """
    Creates a list of unique filenames from a list of input paths.

    This function takes a list of file paths and generates a list of unique
    filenames. It handles potential filename collisions by incrementing a counter
    and appending it to the filename until a unique name is found.

    Args:
        path_inputs (list of Path): A list of file paths.

    Returns:
        list of Path: A list of unique filenames.

    Raises:
        RuntimeError: If a unique filename cannot be found after multiple attempts.
    """

    ret = []

    for item in path_inputs:
        new = item

        # Increment possible file names.
        for i in range(1, 1000):
            # There are several possibilities, we need to catch:
            # 1. A file with the given name already exists.
            # 2. There is already such a named file in `ret`.
            if new in ret or new.exists():
                new = _build_inc_filename(item, i)
            else:
                ret.append(new)
                break

        # Check here whether a new filename was found and raise error if not.
        if len(ret) == 0 or new != ret[-1]:
            raise RuntimeError("could not find an unique filename", new)

    return ret


def _build_inc_filename(path_input: Path, inc: int) -> Path:
    """
    Builds a new path with filename increment.

    This function constructs a new path by taking an input path and a given
    increment as filename addition. The original path and suffix is preserved.

    Args:
        path_input (Path): The original path to build upon.
        inc (int): An integer used to create the incremented filename.

    Returns:
        A new path representing the updated file path.
    """

    path_output = path_input.parent / f"{path_input.stem}_{inc}{path_input.suffix}"
    return path_output


def ms_to_str(milliseconds: int, include_ms: bool = False) -> str:
    """
    Convert milliseconds to a formatted timestamp string in "HH:MM:SS" format.

    Args:
        milliseconds (float): The number of milliseconds to convert.
        include_ms (bool, optional): Whether to include milliseconds in the
        output ("HH:MM:SS.mmm"). Defaults to False.

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


def ms_to_webvtt(milliseconds: int) -> str:
    """
    Converts milliseconds to a WebVTT timestamp string (HH:MM:SS.mmm).

    Args:
        milliseconds: The time in milliseconds.

    Returns:
        A string representing the timestamp in the format HH:MM:SS.mmm.
    """

    return ms_to_str(milliseconds, include_ms=True)


def html_to_text(html_str: str, use_only_body=False) -> str:
    """
    Recursively extracts text content from an HTML node and its children.

    Args:
        node: An HTML node to extract text from.

    Returns:
        A string containing the extracted text.
    """

    # Define the parser class. See
    # https://stackoverflow.com/questions/14694482/converting-html-to-text-with-python
    # for more information on this approach.
    #
    # TODO: move this function and class to its own module.
    class MyHTMLParser(html.parser.HTMLParser):
        def __init__(self, use_only_body=False):
            super().__init__()
            self.result = []
            self.body_found = not use_only_body

        def handle_starttag(self, tag, attrs):
            if self.body_found:
                if tag in HTML_BLOCK_LEVEL_ELEMENTS or tag == "br":
                    self.result.append("\n")
            else:
                if tag == "body":
                    self.body_found = True

        def handle_endtag(self, tag):
            if self.body_found:
                if tag in HTML_BLOCK_LEVEL_ELEMENTS:
                    self.result.append("\n")

        def handle_data(self, data):
            if self.body_found:
                text = html.unescape(data).strip()
                if text:
                    self.result.append(text)

        def get_text(self):
            ret = ""

            for item in self.result:
                if ret and not ret[-1].isspace() and not item.isspace():
                    ret += " "

                ret += item

            return ret.strip()

    parser = MyHTMLParser(use_only_body)
    parser.feed(html_str)

    return parser.get_text()
