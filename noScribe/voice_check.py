"""Check the speaker of each passage against the voice itself.

A transcript segment gets the speaker whose diarization turns overlap it most.
That goes wrong in two ways: a segment that straddles a turn hands the shorter
half to the wrong speaker, and the diarization itself mislabels stretches. Both
are repaired here by a second opinion the diarization never had -- the speaker
embedding of exactly the audio a passage of *text* covers, compared with the
diarization's own speaker centroids.

    units      a segment is cut where a speaker can change: after a sentence
               end, or at a pause of UNIT_PAUSE_S between two words
    voice      one embedding per unit, from pyannote's own model (a second,
               short call to pyannote_mp_worker once the transcript exists)
    decision   a unit moves to another speaker when its voice is closer to that
               speaker's centroid than to the current one: by MARGIN_AGREE when
               the diarization's turns inside the unit name that speaker too, by
               MARGIN_OVERRULE when the voice stands alone

Measured against ground truth at word level (does each word carry the right
speaker?), thresholds chosen on 16 AMI meetings, 16 CallHome German calls and
two spliced conversations, then frozen and checked on material never looked at
before -- 7 more AMI meetings, CallHome calls and 24 VoxConverse recordings:

                                         wrong words   repaired / broken
    faster-whisper   152 000 words       8036 -> 4358     3867 / 189   (95 % right)
    a second engine  125 000 words       6701 -> 3164     3762 / 225   (94 % right)

faster-whisper ran on 61 calls in German, English, Spanish, Japanese and
Mandarin; the second engine (Voxtral, which is not part of this repository and
knows neither Japanese nor Mandarin) on 45. Every language and every corpus is
a net gain, between 88 % and 99.6 % right per corpus with faster-whisper and
between 92 % and 98 % with the other. What the rule breaks sits in a few
recordings whose diarization is badly off to begin with, and those still come
out ahead. Overlapping speech gains too: in passages that are mostly overlapped,
833 / 37 with faster-whisper on the 50 recordings this was counted on.

What was measured and left out, because it added nothing worth its weight:
cutting at speaker changes *without* the voice (on the same pool it repairs
1482 and breaks 310 words with faster-whisper, 83 % right, and ranges from 9 : 1
down to net harm -- 42 / 31 on the Spanish calls, 22 / 39 on four of the tuning
meetings; the cut exposes diarization errors that a whole segment's majority
smooths away); assigning every word on its own, with or without smoothing (cuts
inside a phrase); smoothing the decisions over neighbouring units (a
conversation is not sticky: it broke more than it repaired); a minimum
duration, thresholds that depend on duration or on the diarization's share,
centroids rebuilt from long clean turns, pyannote's exclusive diarization or its
soft activations as the basis, CTC word stamps for Whisper, a guard for similar
voices, and separate thresholds per engine (the optimum is shared).
Scoring in the pipeline's PLDA space (as a log-likelihood ratio or as a cosine)
reaches the same repairs at the same precision and no more; demanding that a
wider crop agrees lowers both; centroids rebuilt once from the confidently
scored units repair about 4 % more words -- real, and not worth a second pass.

The units themselves are not the limit: with perfect labels per unit 0.6 % of
faster-whisper's words would still be wrong, against 5.3 % today and 2.9 % with
this rule (0.8 %, 5.4 % and 2.6 % for the other engine). Finding the changes
*inside* a unit could therefore remove 300 more wrong words at the very most
(444 for the other engine), with a perfect detector, and was not built.
"""

# Word endings that close a sentence, ignoring trailing quotes and brackets.
# Deliberately no colon: a colon lands mid-utterance ("And then he said: ...").
_SENTENCE_END = ('.', '!', '?', '…', '。', '！', '？')
_SENTENCE_TRAIL = '"\'»)] '

# A pause between two words that ends a unit even without punctuation. 0.3 s and
# 0.8 s score within a point of this; 0.25 s cut inside phrases on real speech.
UNIT_PAUSE_S = 0.5

# Cosine margins, a trade of repairs against precision without a sharp optimum.
# On the unseen pool with faster-whisper: 0.0 / 0.25 repairs 3959 and breaks 260
# (94 % right), 0.1 / 0.3 repairs 3408 and breaks 146 (96 %), 0.2 / 0.4 repairs
# 2349 and breaks 63 (97 %). The middle pair keeps the broken words rare enough
# to be worth it everywhere; the same pair is near the best for both engines.
MARGIN_AGREE = 0.1
MARGIN_OVERRULE = 0.3

# Every recording has its own scale of margins. Two similar voices on one channel
# squeeze them: in a studio podcast of two women the centroids had a cosine of
# 0.63, clean passages reached only +-0.3, and three plainly misattributed ones
# sat at 0.19 to 0.25 -- under a margin that recording could hardly ever reach.
# So both margins shrink with the recording's typical margin (the median by which
# passages of UNIT_SCALE_MIN_S or more favour their own speaker), relative to
# MARGIN_SCALE_REF, and never below MARGIN_SCALE_FLOOR of their value. Across the
# test recordings that typical margin runs from 0.24 to 0.86, median 0.5. The
# reference was chosen on the tuning pool (0.45 to 0.7 tried) and checked on the
# unseen one: faster-whisper 4774 -> 4387 wrong words left (3836 repaired, 187
# broken), the second engine 3699 -> 3246 (3672 / 217), precision unchanged --
# where lowering the margins for every recording alike cost two points of it.
MARGIN_SCALE_REF = 0.5
MARGIN_SCALE_FLOOR = 0.5
UNIT_SCALE_MIN_S = 1.5

# A unit that lies wholly inside ONE other speaker's turn (TURN_COVERS of it, and
# no other turn touching it) is as clear a case as the diarization has -- and
# often too short to have much of a voice: a "Perfekt." of 0.3 s between two
# speakers scored +0.03, sat inside the right speaker's turn, and stayed with the
# wrong one because the voice had not confirmed the move. There the voice only
# has to not object. On the unseen pool that repairs 31 more words and breaks 2
# with faster-whisper, 90 more and 8 with the second engine, precision unchanged;
# letting the voice merely not object for EVERY short unit broke as many as it
# repaired.
MARGIN_OBJECT = -0.05
TURN_COVERS = 0.9


def split_units(words):
    """Group a segment's words into units. Returns a list of word lists.

    Words without both time stamps give no span to take a voice from: the
    segment then has no units and stays as it is.
    """
    if not words or any(w.get('start') is None or w.get('end') is None for w in words):
        return []
    units, current = [], []
    for word, following in zip(words, words[1:] + [None]):
        current.append(word)
        token = (word.get('word') or '').rstrip(_SENTENCE_TRAIL)
        if following is None or token.endswith(_SENTENCE_END) or (
                following['start'] - word['end'] >= UNIT_PAUSE_S):
            units.append(current)
            current = []
    return units


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else None


def _scores(embedding, centroids):
    """{label: cosine between the voice and that speaker's centroid}."""
    scores = {}
    for label, centroid in centroids.items():
        score = _cosine(embedding, centroid) if embedding else None
        if score is not None:
            scores[label] = score
    return scores


def margin_scale(units, centroids):
    """By how much the margins of this recording are to shrink (see MARGIN_SCALE_REF).

    units is [(label, embedding, seconds)]. Without a long enough passage that
    favours its own speaker there is nothing to go by, and the margins stay.
    """
    own = []
    for label, embedding, seconds in units:
        scores = _scores(embedding, centroids)
        if seconds >= UNIT_SCALE_MIN_S and label in scores and len(scores) > 1:
            own.append(scores[label] - max(v for k, v in scores.items() if k != label))
    own = sorted(m for m in own if m > 0)
    if not own:
        return 1.0
    middle = len(own) // 2
    typical = own[middle] if len(own) % 2 else (own[middle - 1] + own[middle]) / 2
    return max(MARGIN_SCALE_FLOOR, min(1.0, typical / MARGIN_SCALE_REF))


def decide(current, turns_ms, embedding, centroids, scale=1.0, unit_ms=None):
    """The speaker label a unit should carry.

    current      label the unit has now (its segment's speaker)
    turns_ms     {label: milliseconds of that label's turns inside the unit};
                 empty when the diarization has nothing to add
    embedding    the unit's voice, or None when it could not be computed
    centroids    {label: centroid}
    scale        what margin_scale() found for this recording
    unit_ms      the unit's length, to tell whether one turn covers it (TURN_COVERS)
    """
    scores = _scores(embedding, centroids)
    if current not in scores:
        return current
    label = current
    if turns_ms:
        named = max(turns_ms, key=turns_ms.get)
        covered = unit_ms and len(turns_ms) == 1 and turns_ms[named] >= TURN_COVERS * unit_ms
        needed = MARGIN_OBJECT if covered else MARGIN_AGREE * scale
        if named != label and named in scores and scores[named] - scores[label] > needed:
            label = named
    best = max(scores, key=scores.get)
    if best != label and scores[best] - scores[label] > MARGIN_OVERRULE * scale:
        label = best
    return label


def turns_inside(diarization, start_ms, end_ms):
    """Milliseconds of each label's turns inside [start_ms, end_ms]."""
    totals = {}
    for turn in diarization:
        if turn['start'] > end_ms:
            break
        inside = min(turn['end'], end_ms) - max(turn['start'], start_ms)
        if inside > 0:
            totals[turn['label']] = totals.get(turn['label'], 0) + inside
    return totals


def enabled():
    """NOSCRIBE_VOICE_CHECK=0 restores the plain overlap assignment. (main.py
    also honours `voice_check: 'False'` in config.yml, for an app that was not
    started from a shell.)"""
    import os
    return os.environ.get('NOSCRIBE_VOICE_CHECK', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def _passage(segment, words):
    """The part of `segment` that `words` cover, as a segment of its own.

    The text is rebuilt from the words with the separator the segment itself
    uses: faster-whisper's words carry their own leading space, or none at all
    in languages written without spaces, while another engine may deliver bare
    tokens that its text joins with spaces.
    """
    text = segment.get('text') or ''
    tokens = [w.get('word') or '' for w in segment['words']]
    glued = ''.join(tokens).split() == text.split()
    separator = '' if glued or any(t[:1] in (' ', '\u00a0') for t in tokens) else ' '
    lead = text[:len(text) - len(text.lstrip())]  # a segment's text opens with a space, or not
    return {'start': words[0]['start'], 'end': words[-1]['end'], 'words': words,
            'text': lead + separator.join(w.get('word') or '' for w in words).strip()}


def relabel(segments, diarization, centroids, embed):
    """Decide every segment's passages. Returns (passages, moves).

    segments     [(segment dict, speaker it was written under, inherited)] in
                 transcript order. The speaker may carry the '//' overlap
                 marker; inherited says the diarization was silent there and the
                 segment simply went on under its predecessor's speaker
    diarization  [{'start': ms, 'end': ms, 'label': str}], sorted by start
    centroids    {label: centroid}, labels spelled as in `segments`
    embed        callable: [[start_s, end_s]] -> [embedding or None]

    passages is [(segment dict, speaker to write it under)], and writing them
    again reproduces the transcript wherever nothing moved: an untouched segment
    comes back with the speaker it had, marker and all, and an inherited one with
    '' so that it goes on following its predecessor. moves lists every passage
    that ends up under another speaker, as [(passage, speaker before, after)].
    """
    if len(centroids) < 2:
        return [(segment, '' if inherited else written) for segment, written, inherited in segments], []
    plan, spans = [], []
    for segment, written, inherited in segments:
        current = written.lstrip('/')
        units = split_units(segment.get('words')) if current in centroids else []
        plan.append((segment, written, inherited, current, units))
        spans += [[unit[0]['start'], unit[-1]['end']] for unit in units]
    voices = embed(spans) if spans else []
    voices = list(voices) + [None] * (len(spans) - len(voices))
    scale = margin_scale([(current, voice, end - start) for (start, end), voice, current in zip(
        spans, voices, (current for _, _, _, current, units in plan for _ in units))], centroids)
    embeddings = iter(voices)

    passages, moves = [], []
    # Who speaks without the overlap marker -- the turn others talk into -- as the
    # transcript had it, and as it stands after the moves so far; and the speaker
    # the writer is in, which is what an inherited segment goes on under.
    floor_was = floor = state = ''
    for segment, written, inherited, current, units in plan:
        marked = written.startswith('//')
        # An inherited segment is held against the speaker it would inherit now.
        base = state.lstrip('/') if inherited and state.lstrip('/') in centroids else current
        labels = []
        for unit in units:
            # A unit that is its whole segment already had the diarization's say
            # on exactly this span; only the voice alone can move it.
            start_ms, end_ms = round(unit[0]['start'] * 1000), round(unit[-1]['end'] * 1000)
            turns_ms = turns_inside(diarization, start_ms, end_ms) if len(units) > 1 else {}
            labels.append(decide(base, turns_ms, next(embeddings, None), centroids, scale, end_ms - start_ms))
        runs = []
        for unit, label in zip(units, labels):
            if runs and runs[-1][0] == label:
                runs[-1][1].extend(unit)
            else:
                runs.append((label, list(unit)))
        if len(runs) < 2:  # written as one piece: keep the segment itself
            runs = [(labels[0] if labels else base, None)]
        for first, (label, words) in zip([True] + [False] * len(runs), runs):
            if inherited and first and label == base:
                speaker = ''  # goes on following its predecessor, wherever that went
            elif marked and not inherited and not (
                    label == floor and (label != current or floor != floor_was)):
                # Overlapping speech stays marked as such, moved or not -- unless the
                # voice now is the one holding the floor, whose turn it continues.
                speaker = '//' + label
            else:
                speaker = label
            passage = segment if words is None else _passage(segment, words)
            if (speaker or state).lstrip('/') != current:
                moves.append((passage, written, speaker or state))
            passages.append((passage, speaker))
            if speaker:
                state = speaker
                if not speaker.startswith('//'):
                    floor = label
        if not marked:
            floor_was = current
    return passages, moves
