"""Bootstrap confidence intervals for WER and CER on the reference passage.

The reference is 422 words / 2034 characters, so a single character is 0.049 CER
points. That makes it easy to over-read a 0.05-point difference between builds.
This script replaces the rule of thumb ("below ~0.15 points is noise") with a
measured interval, and reports the paired difference between two builds --
which is the quantity that actually decides whether a build change did anything.

Input is the transcripts that docs/scripts/wer.py leaves in /tmp/wer_<build>.txt.

    python docs/scripts/wer.py <reference.txt> <audio.wav> <build> [build ...]
    python docs/scripts/bootstrap_cer.py <reference.txt> <build> [build ...]

Method: align hypothesis to reference with a Levenshtein DP that keeps
backpointers, attribute every edit to the reference position it lands on, bucket
those positions into blocks, then resample blocks with replacement. Blocks
rather than single tokens because errors are correlated within a phrase -- a
token-level bootstrap would understate the interval.
"""
import random
import sys
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, OVERLAP  # same normalisation as the scoring script

RESAMPLES = 10000
WORD_BLOCK = 20      # reference words per bootstrap block
CHAR_BLOCK = 100     # reference characters per bootstrap block
SEED = 20260727      # fixed: the interval must not move between runs


def align_costs(ref, hyp):
    """Edit count attributed to each reference index.

    Returns a list `cost` of len(ref): cost[i] is how many edits the cheapest
    alignment charges at reference position i. Insertions are charged to the
    reference position they precede, so every edit lands somewhere in the
    reference and the totals still sum to the edit distance.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = distance; move[i][j] in {"eq","sub","del","ins"}
    d = [[0] * (m + 1) for _ in range(n + 1)]
    mv = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
        mv[i][0] = "del"
    for j in range(1, m + 1):
        d[0][j] = j
        mv[0][j] = "ins"
    for i in range(1, n + 1):
        ri = ref[i - 1]
        di, dp = d[i], d[i - 1]
        mi = mv[i]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                di[j] = dp[j - 1]
                mi[j] = "eq"
            else:
                sub, dele, ins = dp[j - 1], dp[j], di[j - 1]
                best = min(sub, dele, ins)
                di[j] = best + 1
                mi[j] = "sub" if best == sub else ("del" if best == dele else "ins")
    cost = [0] * max(1, n)
    i, j = n, m
    while i > 0 or j > 0:
        step = mv[i][j]
        if step == "eq":
            i, j = i - 1, j - 1
        elif step == "sub":
            cost[i - 1] += 1
            i, j = i - 1, j - 1
        elif step == "del":
            cost[i - 1] += 1
            i -= 1
        else:  # insertion: charge it to the reference position it sits before
            cost[min(i, n - 1)] += 1
            j -= 1
    return cost


def blocks(cost, size):
    """(errors, reference units) per block."""
    out = []
    for s in range(0, len(cost), size):
        chunk = cost[s:s + size]
        out.append((sum(chunk), len(chunk)))
    return out


def rate(blks):
    """The observed error rate over the unsampled blocks, in percent.

    This is the point estimate. It is NOT the middle of the bootstrap interval:
    the resampling distribution can sit off-centre, so the midpoint is a
    different quantity and would disagree with the per-build numbers.
    """
    e = sum(x for x, _ in blks)
    u = sum(x for _, x in blks)
    return e / u * 100 if u else 0.0


def boot(blks, rng):
    """Bootstrap distribution of the error rate over blocks."""
    n = len(blks)
    dist = []
    for _ in range(RESAMPLES):
        e = u = 0
        for _ in range(n):
            be, bu = blks[rng.randrange(n)]
            e += be
            u += bu
        dist.append(e / u * 100 if u else 0.0)
    dist.sort()
    return dist


def ci(dist):
    lo = dist[int(0.025 * len(dist))]
    hi = dist[int(0.975 * len(dist)) - 1]
    return lo, hi


def paired(blks_a, blks_b, rng):
    """Bootstrap the DIFFERENCE a-b on the same resampled blocks.

    Paired, because both builds are scored on the same passage: the shared
    difficulty of the material cancels and the interval is far tighter than
    comparing two independent intervals.
    """
    n = len(blks_a)
    dist = []
    for _ in range(RESAMPLES):
        ea = ua = eb = ub = 0
        for _ in range(n):
            k = rng.randrange(n)
            ea += blks_a[k][0]; ua += blks_a[k][1]
            eb += blks_b[k][0]; ub += blks_b[k][1]
        dist.append((ea / ua - eb / ub) * 100 if ua and ub else 0.0)
    dist.sort()
    return dist


raw = open(sys.argv[1], encoding="utf-8").read()
ref_words = norm(OVERLAP.sub(" ", raw))
ref_chars = list("".join(ref_words))
names = [p.rstrip("/").split("/")[-1] for p in sys.argv[2:]]

print(f"# Reference {len(ref_words)} words / {len(ref_chars)} chars")
print(f"# {RESAMPLES} resamples, blocks: {WORD_BLOCK} words / {CHAR_BLOCK} chars, seed {SEED}\n")

table = {}
for name in names:
    path = f"/tmp/wer_{name}.txt"
    try:
        hyp_words = norm(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        print(f"!! {path} missing -- run docs/scripts/wer.py first")
        continue
    hyp_chars = list("".join(hyp_words))
    wb = blocks(align_costs(ref_words, hyp_words), WORD_BLOCK)
    cb = blocks(align_costs(ref_chars, hyp_chars), CHAR_BLOCK)
    table[name] = (wb, cb)
    rng = random.Random(SEED)
    wd, cd = boot(wb, rng), boot(cb, rng)
    wl, wh = ci(wd)
    cl, ch = ci(cd)
    print(f"{name:32s} WER {rate(wb):5.2f}%  [{wl:5.2f}, {wh:5.2f}]"
          f"   CER {rate(cb):5.2f}%  [{cl:5.2f}, {ch:5.2f}]")

if len(table) > 1:
    print("\nPaired differences (95% interval; if it contains 0, the "
          "difference is not demonstrable):")
    keys = list(table)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            rng = random.Random(SEED)
            wd = paired(table[ka][0], table[kb][0], rng)
            cd = paired(table[ka][1], table[kb][1], rng)
            wl, wh = ci(wd)
            cl, ch = ci(cd)
            wsig = " " if wl <= 0 <= wh else "*"
            csig = " " if cl <= 0 <= ch else "*"
            print(f"  {ka} - {kb}")
            dw = rate(table[ka][0]) - rate(table[kb][0])
            dc = rate(table[ka][1]) - rate(table[kb][1])
            print(f"      dWER {dw:+5.2f}  [{wl:+5.2f}, {wh:+5.2f}] {wsig}"
                  f"   dCER {dc:+5.2f}  [{cl:+5.2f}, {ch:+5.2f}] {csig}")
    print("\n* = interval excludes 0")
