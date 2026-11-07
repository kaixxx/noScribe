"""Decide whether the current GitHub Actions ref may publish the latest image."""

import os
import re
import subprocess


def stable_release_key(tag):
    """Compare numeric release versions and optional Docker image revisions."""
    match = re.fullmatch(r"v([0-9]+(?:\.[0-9]+)*)(?:-r([0-9]+))?", tag)
    if match is None:
        return None
    version = list(map(int, match[1].split(".")))
    while len(version) > 1 and version[-1] == 0:
        version.pop()
    return tuple(version), int(match[2] or 0)


def is_latest_release(ref, tags):
    if not ref.startswith("refs/tags/"):
        return False
    tag = ref.removeprefix("refs/tags/")
    current = stable_release_key(tag)
    if current is None or tag not in tags:
        return False
    stable = [key for name in tags if (key := stable_release_key(name)) is not None]
    return current == max(stable)


def main():
    ref = os.environ["GITHUB_REF"]
    latest = False
    if ref.startswith("refs/tags/") and stable_release_key(ref.removeprefix("refs/tags/")) is not None:
        # Read remote tags so shallow checkouts and long builds use current data.
        refs = subprocess.check_output(
            ["git", "ls-remote", "--tags", "--refs", "origin", "refs/tags/v*"],
            text=True,
        )
        tags = [line.split("\t", 1)[1].removeprefix("refs/tags/")
                for line in refs.splitlines()]
        latest = is_latest_release(ref, tags)
    print(f"latest={str(latest).lower()}")


if __name__ == "__main__":
    main()
