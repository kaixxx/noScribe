"""Show every spot where the three mini builds disagree, with context.

Reference is bf16 (the unquantised release). Both quantised builds are aligned
against it; any region where at least one differs is printed as a three-line
block, so the *kind* of difference is visible, not just a percentage.

    python docs/scripts/three_way_diff.py <dir with mx_voxtral-mini-*.txt>
"""
import difflib, sys

d = sys.argv[1].rstrip('/')
ref = open(f"{d}/mx_voxtral-mini-bf16.txt").read().split()
builds = {n: open(f"{d}/mx_voxtral-mini-{n}.txt").read().split()
          for n in ("8bit", "8bit-dense")}

# For each build: map every bf16 word index -> that build's text for the region.
CTX = 4


def regions(other):
    """List of (ref_start, ref_end, other_words) for non-equal opcodes."""
    sm = difflib.SequenceMatcher(None, ref, other)
    return [(i1, i2, other[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag != "equal"]


regs = {n: regions(w) for n, w in builds.items()}

# Merge all divergence spans on the bf16 axis, then widen with context.
spans = sorted({(i1, i2) for r in regs.values() for i1, i2, _ in r})
merged = []
for i1, i2 in spans:
    if merged and i1 <= merged[-1][1] + 2 * CTX:
        merged[-1][1] = max(merged[-1][1], i2)
    else:
        merged.append([i1, i2])

print(f"# {len(merged)} divergence spots on {len(ref)} bf16 words\n")
for k, (i1, i2) in enumerate(merged, 1):
    lo, hi = max(0, i1 - CTX), min(len(ref), i2 + CTX)
    print(f"--- Stelle {k} (~Wort {i1}) ---")
    print(f"  bf16 : ...{' '.join(ref[lo:hi])}...")
    for n, w in builds.items():
        # rebuild this build's words for the bf16 window from its opcodes
        sm = difflib.SequenceMatcher(None, ref, w)
        out = []
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if a2 <= lo or a1 >= hi:
                continue
            if tag == "equal":
                s, e = max(a1, lo), min(a2, hi)
                out.extend(w[b1 + (s - a1):b1 + (e - a1)])
            else:
                out.extend(w[b1:b2])
        same = " (= bf16)" if out == ref[lo:hi] else ""
        print(f"  {n:5s}: ...{' '.join(out)}...{same}")
    print()
