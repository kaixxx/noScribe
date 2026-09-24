"""List the spots where two arms hear differently -- with a timestamp.

Why: a reference produced by correcting ONE transcript leans towards that
transcript. Errors overlooked while correcting stand in the reference as truth
afterwards -- and favour exactly the arm the draft came from. Every other arm
is penalised for the same spot even when it hears it correctly.

That cannot be computed away, but it can be settled cheaply: there are only a
handful of spots where the arms diverge at all. This script lists them with an
approximate timestamp and what the reference says there, so that one can
listen to those spots specifically instead of the whole passage again.

The timestamp is an estimate: word position times mean word duration. It
points to the neighbourhood, not the second.

The arm transcripts are read from /tmp/wer_<arm>.txt, the scratch files that
wer.py, preproc_cer.py or pair_cer.py leave behind.

    python docs/scripts/adjudicate.py <reference.txt> <seconds> <arm_a> <arm_b> [arm ...]
"""
import difflib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, OVERLAP

CONTEXT = 4


def load(label):
    return norm(open(f"/tmp/wer_{label}.txt", encoding="utf-8").read())


def stamp(pos, total, dur):
    s = int(pos / max(1, total) * dur)
    return f"{s//60:02d}:{s%60:02d}"


def main():
    ref_path, dur = sys.argv[1], float(sys.argv[2])
    a_label, others = sys.argv[3], sys.argv[4:]
    raw = open(ref_path, encoding="utf-8").read()
    ref = norm(OVERLAP.sub(" ", raw))
    a = load(a_label)

    for b_label in others:
        b = load(b_label)
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        print(f"\n{'='*74}\n{a_label}  vs  {b_label}: {len(ops)} spots\n")
        for tag, i1, i2, j1, j2 in ops:
            # Where is the spot in the reference? Searched via the shared
            # context before it, so the reference words can stand alongside.
            pre = a[max(0, i1 - CONTEXT):i1]
            k = -1
            for s in range(len(ref) - len(pre) + 1):
                if pre and ref[s:s + len(pre)] == pre:
                    k = s + len(pre)
                    break
            ref_span = " ".join(ref[k:k + max(1, i2 - i1)]) if k >= 0 else "?"
            print(f"  ~{stamp(i1, len(a), dur)}  …{' '.join(pre)} ▸")
            print(f"      {a_label:14s} {' '.join(a[i1:i2]) or '(nothing)'!r}")
            print(f"      {b_label:14s} {' '.join(b[j1:j2]) or '(nothing)'!r}")
            print(f"      {'REFERENCE':14s} {ref_span!r}")
            print(f"      then: {' '.join(a[i2:i2+CONTEXT])}")
    print("\nWhich one is right is for the ear to decide. Where the reference follows")
    print("the first arm although the second is right, the reference leans towards")
    print("the first -- and the comparison is biased in its favour.")


if __name__ == "__main__":
    main()
