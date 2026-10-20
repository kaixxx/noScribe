"""The installed Voxtral stack is the one the engine was validated against.

`environments/requirements_voxtral_macOS_arm64.txt` pins mlx, mlx-lm and
mlx-voxtral to exact versions, and every measurement in VOXTRAL.md and docs/
describes those versions -- but nothing used to check what is actually in the
environment. That gap is not cosmetic: mlx-voxtral 0.0.5 computes the log-Mel
per 30 s chunk and 0.0.6 over the whole audio, so a developer left on the older
pin feeds the encoder a different input from the one every documented number was
measured on, and no test says a word. The smoke test cannot catch it either -- it
compares the two decode paths of whatever happens to be installed.

The version checks skip where the optional Apple-Silicon stack is absent, which
is how Linux CI and every non-Voxtral checkout see them; the check that the
pins are still exact is a plain file read and runs everywhere.
"""
import pathlib
import re

import pytest

from importlib.metadata import PackageNotFoundError, version

REQUIREMENTS = (pathlib.Path(__file__).resolve().parent.parent
                / "environments" / "requirements_voxtral_macOS_arm64.txt")

PIN = re.compile(r"^([A-Za-z0-9._-]+)==([A-Za-z0-9.]+)\s*$")


def _pins():
    """{package: version} for the `name==version` lines, comments ignored."""
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        m = PIN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_the_requirements_file_actually_pins_the_stack():
    """A guard on the guard: if the pins were loosened to `>=`, the test below
    would pass vacuously by having nothing left to compare."""
    pins = _pins()
    assert {"mlx", "mlx-lm", "mlx-voxtral"} <= set(pins), \
        f"exact pins missing from {REQUIREMENTS.name}: found {sorted(pins)}"


@pytest.mark.parametrize("package", ["mlx", "mlx-lm", "mlx-voxtral"])
def test_the_installed_version_matches_the_pin(package):
    pinned = _pins()[package]
    try:
        installed = version(package)
    except PackageNotFoundError:
        pytest.skip(f"{package} not installed (optional Apple-Silicon stack)")
    assert installed == pinned, (
        f"{package} {installed} installed, {REQUIREMENTS.name} pins {pinned}. "
        f"The engine and its measurements were validated against the pin; "
        f"re-run `pip install -r environments/{REQUIREMENTS.name}`.")
