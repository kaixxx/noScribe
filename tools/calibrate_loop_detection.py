"""Re-measure the repetition-loop thresholds against real noScribe transcripts.

The thresholds in voxtral_engine (DEGENERATE_CYCLE_REPEATS, and the
DEGENERATE_COMPRESSION_RATIO net behind it) are not guesses -- they sit in the
gap between two measured populations: what clean German transcripts do, and
what a pass that collapsed into a repetition loop does. That gap has to be
re-checked whenever a threshold moves or a new model lands, and the only honest
source for it is finished transcripts.

Those live in noScribe's own log files, which record the transcript as it is
written. They contain real interview material, so nothing here is committed:
the script reads the log directory in place and prints numbers.

    python tools/calibrate_loop_detection.py                # default log dir
    python tools/calibrate_loop_detection.py --logs PATH    # somewhere else
    python tools/calibrate_loop_detection.py --list         # per-chunk detail

A threshold is well-placed when "highest clean" and "lowest flagged" stay far
apart. If a clean chunk creeps up towards the threshold, that is a false
positive waiting to happen -- look at the chunk before raising the number,
because a genuine loop of a new shape looks the same from here.
"""
import argparse
import os
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noScribe.voxtral_engine import (  # noqa: E402
    DEGENERATE_COMPRESSION_RATIO,
    DEGENERATE_CYCLE_REPEATS,
    _longest_cycle_repeats,
    _looks_degenerate,
)

CHUNK_MARKER = re.compile(r"Transcribing chunk (\d+)/")
# The GUI writes these into the log around the transcript; they are not model
# output and would otherwise show up as a repeating cycle of their own.
GUI_TIMESTAMP = re.compile(r"\[\d\d:\d\d:\d\d\]")
MIN_WORDS = 200  # shorter blocks are not a representative pass


def short(name, width=40):
    """Keep the END of a long log name: transcripts of the same series differ
    in their suffix ("... day_1" / "... day_2"), so cutting the tail makes two
    different files look like one."""
    return name if len(name) <= width else "\u2026" + name[-(width - 1):]


def default_log_dir():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/noScribe/log"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", "")) / "noScribe/log"
    return Path.home() / ".config/noScribe/log"


def chunks(log_dir):
    """Yield (log name, chunk number, transcript text) per transcribed chunk."""
    for f in sorted(Path(log_dir).glob("*.log")):
        lines = f.read_text(errors="replace").splitlines()
        marks = [(int(m.group(1)), i) for i, ln in enumerate(lines)
                 if (m := CHUNK_MARKER.search(ln))]
        starts = [i for _, i in marks[1:]] + [len(lines)]
        for (number, start), end in zip(marks, starts):
            text = GUI_TIMESTAMP.sub(" ", "\n".join(lines[start:end]))
            if len(text.split()) >= MIN_WORDS:
                yield f.name, number, text


def cycle_sample(words, max_k=8):
    """The words behind the count -- so a flagged chunk can be eyeballed
    without opening the log. The count itself always comes from the engine's
    own _longest_cycle_repeats, so the two cannot drift apart."""
    best = (0, 0, 0)
    for k in range(1, max_k + 1):
        run = 0
        for i in range(k, len(words)):
            run = run + 1 if words[i] == words[i - k] else 0
            if run > best[0]:
                best = (run, k, i)
    run, _, end = best
    return " ".join(words[end - run:end])[:60]


def compression_ratio(text):
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=default_log_dir(),
                    help="directory holding noScribe's log files (default: the app's log dir)")
    ap.add_argument("--list", action="store_true", help="one line per chunk")
    args = ap.parse_args()

    if not args.logs.is_dir():
        sys.exit(f"No log directory at {args.logs} -- pass --logs PATH.")

    clean, flagged = [], []
    for name, number, text in chunks(args.logs):
        words = text.split()
        repeats = _longest_cycle_repeats(words)
        sample = cycle_sample(words)
        row = (repeats, compression_ratio(text), len(words), name, number, sample)
        (flagged if _looks_degenerate(text) else clean).append(row)
        if args.list:
            mark = "LOOP" if _looks_degenerate(text) else "    "
            print(f"{mark} {short(name):40} chunk {number:2}  "
                  f"{len(words):5} words  cycle x{repeats:<4} ratio {row[1]:.2f}")

    if not clean and not flagged:
        sys.exit(f"No transcribed chunks found in {args.logs}.")

    if args.list:
        print()
    print(f"chunks measured : {len(clean) + len(flagged)}  from {args.logs}")
    print(f"thresholds      : cycle repeats >= {DEGENERATE_CYCLE_REPEATS}, "
          f"compression ratio > {DEGENERATE_COMPRESSION_RATIO}")
    print()

    if clean:
        top = max(clean)
        print(f"clean   : {len(clean):3} chunks, highest cycle count {top[0]} "
              f"({short(top[3])} chunk {top[4]})")
    if flagged:
        low = min(flagged)
        print(f"flagged : {len(flagged):3} chunks, lowest cycle count  {low[0]} "
              f"({short(low[3])} chunk {low[4]})")
        for repeats, _, _, name, number, sample in sorted(flagged, reverse=True):
            print(f"          x{repeats:<4} {short(name):40} chunk {number:2}  {sample!r}")

    print()
    if clean and flagged:
        gap = min(f[0] for f in flagged) - max(c[0] for c in clean)
        print(f"gap between the populations: {gap} repeats "
              f"(threshold {DEGENERATE_CYCLE_REPEATS} sits inside it)"
              if gap > 0 else
              "POPULATIONS OVERLAP -- a clean chunk cycles as often as a loop; "
              "the cycle count alone cannot separate them any more.")
    elif not flagged:
        print("Nothing flagged. Either the corpus is clean or the threshold is "
              "too high -- check the highest clean count above.")


if __name__ == "__main__":
    main()
