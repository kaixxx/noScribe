"""Reference-free paired diff between two builds over hours of real audio.

The question is whether the audio encoder's precision changes what the model
HEARS or only how it SPELLS. A hand-corrected reference answers that on two
minutes of audio; this answers it on hours, without needing one.

Method: transcribe the same audio with both builds in identical fixed windows,
then classify every difference between the two transcripts by how similar the
differing spans are at character level:

  orthographic  short span, characters nearly the same   ("Vita Flor"/"VitaFlor")
  lexical       characters differ                        (a different word was heard)
  cascade       long span -- under greedy decoding one flipped token makes the
                text diverge for a while; that is not an encoder effect and is
                counted separately rather than as a mishearing

If the encoder's bit width only produces orthographic differences, its precision
is not buying recognition quality.

    python docs/scripts/encoder_diff.py <reference-build> <build> [build ...] \
        --audio <file> [file ...]

Every build is compared against the first one. Each build transcribes each file
exactly once, so adding a third build costs one more pass, not three.

Windows are cut at a fixed length so both builds see byte-identical input; no
pause-alignment, no retry ladder, nothing that could differ between runs.
"""
import os
import sys
import json
import pathlib
import tempfile
import time
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from noScribe.voxtral_engine import _Voxtral
from wer import norm

WINDOW_SEC = 600      # fits mini8 on 32 GB with room to spare (~10 GB peak)
CASCADE_WORDS = 8     # longer differing spans are greedy divergence, not hearing
ORTHO_SIM = 0.80      # character similarity at or above this = same word, spelled differently
LEXICAL_SIM = 0.50    # below this = a different word


def load_16k(path):
    """Mono 16 kHz float32, the way the engine feeds the model."""
    import av
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    parts = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            parts.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        parts.append(out.to_ndarray().reshape(-1))
    return np.concatenate(parts).astype("float32") if parts else np.zeros(0, "float32")


def transcribe_windows(build, audio, cache):
    """Transcribe fixed windows, caching per (build, window) so a re-run is free."""
    key = str(build)
    if key in cache:
        return cache[key]
    vox = _Voxtral(build)
    out = []
    step = WINDOW_SEC * 16000
    for w, start in enumerate(range(0, len(audio), step)):
        seg = audio[start:start + step]
        if len(seg) < 16000:          # ignore a sub-second tail
            continue
        dur = len(seg) / 16000
        t0 = time.time()
        out.append(vox.transcribe_array(seg, "de", max_new_tokens=int(dur * 20) + 512))
        print(f"    {key.split('/')[-1]:28s} Fenster {w + 1} "
              f"({dur:.0f}s) in {time.time() - t0:.0f}s", flush=True)
    del vox
    mx.clear_cache()
    cache[key] = out
    return out


def classify(a_span, b_span):
    n = max(len(a_span), len(b_span))
    if n > CASCADE_WORDS:
        return "cascade", 0.0
    if not a_span or not b_span:
        # One side emitted nothing. That is not a mishearing of some other
        # word, it is an omission -- a different failure, and one that matters:
        # a build that drops words is worse even if it never misspells.
        return "omission", 0.0
    sim = SequenceMatcher(None, "".join(a_span), "".join(b_span)).ratio()
    if sim >= ORTHO_SIM:
        return "orthographic", sim
    if sim < LEXICAL_SIM:
        return "lexical", sim
    return "ambiguous", sim


def compare(text_a, text_b):
    a, b = norm(text_a), norm(text_b)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    events = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a_span, b_span = a[i1:i2], b[j1:j2]
        kind, sim = classify(a_span, b_span)
        events.append({"kind": kind, "sim": round(sim, 3),
                       "a": " ".join(a_span), "b": " ".join(b_span)})
    return events, len(a), len(b)


split = sys.argv.index("--audio")
builds = sys.argv[1:split]
files = sys.argv[split + 1:]
short = [b.rstrip("/").split("/")[-1] for b in builds]
ref_build, ref_name = builds[0], short[0]

# results[other_build] = (events, words_ref, words_other)
results = {b: ([], 0, 0) for b in builds[1:]}
for path in files:
    print(f"\n== {path}", flush=True)
    audio = load_16k(path)
    print(f"   {len(audio) / 16000 / 60:.1f} min, "
          f"{-(-len(audio) // (WINDOW_SEC * 16000))} Fenster", flush=True)
    cache = {}
    wins = {b: transcribe_windows(b, audio, cache) for b in builds}
    for b in builds[1:]:
        ev, wr, wo = results[b]
        for ta, tb in zip(wins[ref_build], wins[b]):
            e, na, nb = compare(ta, tb)
            ev += e
            wr += na
            wo += nb
        results[b] = (ev, wr, wo)

ACOUSTIC = ("lexical", "omission")
dump = {"reference": ref_name, "files": files, "window_sec": WINDOW_SEC,
        "thresholds": {"cascade_words": CASCADE_WORDS, "ortho_sim": ORTHO_SIM,
                       "lexical_sim": LEXICAL_SIM},
        "comparisons": {}}

for b in builds[1:]:
    events, words_r, words_o = results[b]
    name_b = b.rstrip("/").split("/")[-1]
    counts = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    decisive = sum(counts.get(k, 0) for k in ("orthographic", "ambiguous") + ACOUSTIC)
    acoustic = sum(counts.get(k, 0) for k in ACOUSTIC)

    print(f"\n{'=' * 72}")
    print(f"{ref_name}  vs  {name_b}")
    print(f"{words_r} / {words_o} words, {len(events)} differing spots\n")
    for kind in ("orthographic", "ambiguous", "lexical", "omission", "cascade"):
        n = counts.get(kind, 0)
        share = (f"{n / decisive * 100:5.1f}% of the classifiable"
                 if decisive and kind != "cascade" else "")
        print(f"  {kind:14s} {n:5d}   {n / max(1, words_r) * 1000:6.2f} per 1000 words   {share}")
    if decisive:
        print(f"\n  -> acoustic (lexical+omission): {acoustic}/{decisive} = "
              f"{acoustic / decisive * 100:.1f}% of the classifiable spots, "
              f"{acoustic / max(1, words_r) * 1000:.2f} per 1000 words")
        print("     If this share is small, the encoder precision only changes the "
              "spelling, not what was heard.")

    print("\n  Beispiele je Klasse (bis zu 6):")
    for kind in ("lexical", "omission", "ambiguous", "orthographic"):
        ex = [e for e in events if e["kind"] == kind][:6]
        if ex:
            print(f"    -- {kind}")
            for e in ex:
                print(f"       sim {e['sim']:.2f}  {ref_name[-5:]}: {e['a'][:55]!r}  "
                      f"{name_b[-5:]}: {e['b'][:55]!r}")

    dump["comparisons"][name_b] = {"words_ref": words_r, "words_other": words_o,
                                   "counts": counts, "events": events}

# NOT into the repository: the event list quotes the audio verbatim, and the
# audio is real interview material. Only the aggregate counts printed above are
# safe to copy into docs/. Override the location with NOSCRIBE_MESS_DIR.
out_dir = pathlib.Path(os.environ.get("NOSCRIBE_MESS_DIR", tempfile.gettempdir()))
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"encoder_diff_{ref_name}.json"
out.write_text(json.dumps(dump, ensure_ascii=False, indent=1))
print(f"\nRohdaten (ausserhalb des Repos): {out}")
