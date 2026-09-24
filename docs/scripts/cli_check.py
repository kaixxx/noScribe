"""Assertions on the three transcript files a noScribe CLI smoke run leaves behind.

Reads /tmp/out_mapped.vtt, /tmp/out_unmapped.html and /tmp/out_short.txt --
scratch files the user produces with three CLI invocations -- and checks the
output-format and speaker-name plumbing: mapped names reach the VTT voices,
S00/S01 survive without --speaker-names, no HTML leaks into the txt path. A
format check, not a measurement; listed under docs/voxtral-benchmarks.md §8.
"""
import os, re, sys

def rd(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None

fails = []

# --- A: mapped VTT ---
vtt = rd("/tmp/out_mapped.vtt")
print("=== A: /tmp/out_mapped.vtt ===")
if not vtt:
    fails.append("A: vtt missing")
else:
    cues = re.findall(r"\d\d:\d\d:\d\d\.\d\d\d --> \d\d:\d\d:\d\d\.\d\d\d\n(.*?)(?:\n\n|\Z)", vtt, re.S)
    voices = re.findall(r"<v ([^>]+)>", vtt)
    texts = [re.sub(r"<v [^>]+>", "", c).strip() for c in cues]
    lens = [len(t) for t in texts if t]
    print(f"  starts WEBVTT: {vtt.startswith('WEBVTT')}  cues: {len(cues)}  voices: {sorted(set(voices))}")
    print(f"  cue text len: max={max(lens) if lens else 0} avg={sum(lens)//len(lens) if lens else 0}")
    for t in texts[:3]:
        print(f"    cue: {t[:70]!r}")
    if not vtt.startswith("WEBVTT"): fails.append("A: no WEBVTT header")
    if len(cues) < 3: fails.append(f"A: too few cues ({len(cues)})")
    if not any(v in ("Mona", "Lena") for v in voices):
        fails.append(f"A: mapped names not in voices {set(voices)}")
    if any(v in ("S00", "S01") for v in voices):
        fails.append(f"A: unmapped S00/S01 leaked into voices {set(voices)}")
    if lens and max(lens) > 300:
        fails.append(f"A: cue too large ({max(lens)} chars) -- not subtitle-sized")

# --- B: unmapped HTML (old S00/S01 behavior) ---
htm = rd("/tmp/out_unmapped.html")
print("\n=== B: /tmp/out_unmapped.html ===")
if not htm:
    fails.append("B: html missing")
else:
    anchors = re.findall(r'name="ts_[\d.]+_[\d.]+_([^"]*)"', htm)
    print(f"  <p> tags: {htm.count('<p')}  ts_ anchors: {len(anchors)}  speakers: {sorted(set(anchors))[:8]}")
    body_speakers = set(a for a in anchors if a)
    if "<p" not in htm: fails.append("B: no <p> paragraphs")
    if len(anchors) < 3: fails.append(f"B: too few ts_ anchors ({len(anchors)})")
    if not any(re.fullmatch(r"S\d\d", s) for s in body_speakers):
        fails.append(f"B: expected S00/S01 labels (old behavior), got {body_speakers}")
    if any(n in htm for n in ("Mona", "Lena")):
        fails.append("B: real names present despite no --speaker-names")

# --- C: short-path TXT ---
txt = rd("/tmp/out_short.txt")
print("\n=== C: /tmp/out_short.txt ===")
if not txt:
    fails.append("C: txt missing")
else:
    words = len(txt.split())
    tags = len(re.findall(r"<[a-zA-Z/][^>]*>", txt))
    print(f"  words: {words}  html-tags: {tags}")
    print(f"    head: {txt[:120]!r}")
    if words < 50: fails.append(f"C: too little text ({words} words)")
    if tags > 0: fails.append(f"C: raw HTML tags leaked into txt ({tags})")

print("\n===== FORMAT/SPEAKER RESULT:", "ALL OK" if not fails else f"FAILS: {fails}", "=====")
sys.exit(0 if not fails else 1)
