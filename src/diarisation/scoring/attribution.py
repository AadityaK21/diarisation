"""Word-level speaker attribution against a diarisation output.

Each reference word is given to the system speaker whose turns overlap it
most (a zero-length word: the speaker active at its time); no overlap = the
word is unassigned. System speakers are mapped to reference speakers in one
of two ways:

  optimal          md-eval's one-to-one mapping over all evaluated frames
                   (permutation-invariant: the WDER-style score).
  majority_single  the brief's fixed mapping: each system speaker goes to the
                   reference speaker it overlaps most on single-speaker
                   frames only (many-to-one allowed), fixed once.

Outcome per word:
  correct              mapped system speaker = the word's speaker
  other_active_missed  credited to another reference speaker talking at that
                       moment, and only one system speaker covered the word:
                       the system missed the word's speaker
  other_active_rule    the same, but two or more system speakers covered the
                       word: the one-speaker-per-word rule picked the wrong one
  inactive             credited to a reference speaker not talking at that moment
  unmapped       the system speaker has no reference counterpart
  unassigned     no system speaker covers the word at all

A word is "in overlap" when >= 2 reference speakers are active at its midpoint.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.ami_words import Word
from ..intervals import count_timeline, per_speaker_union
from ..io.rttm import Segment
from ..io.uem import Region
from .der import frame_arrays, overlap_seconds

OUTCOMES = ("correct", "other_active_missed", "other_active_rule", "inactive", "unmapped", "unassigned")


def _covering(regions: List[Tuple[float, float]], t: float) -> bool:
    i = bisect_right(regions, (t, float("inf"))) - 1
    return i >= 0 and regions[i][0] <= t < regions[i][1]


def _overlap_with(regions: List[Tuple[float, float]], a: float, b: float) -> float:
    i = max(bisect_right(regions, (a, float("inf"))) - 1, 0)
    tot = 0.0
    while i < len(regions) and regions[i][0] < b:
        tot += max(0.0, min(b, regions[i][1]) - max(a, regions[i][0]))
        i += 1
    return tot


def assign_with_count(words: Sequence[Word], hyp: Sequence[Segment]) -> List[Tuple[Optional[str], int]]:
    """(system speaker, number of system speakers overlapping the word) per word.

    Ties in overlap duration go to the label that sorts first: arbitrary but
    deterministic, and exactly the choice a one-speaker-per-word rule has to make.
    """
    by = per_speaker_union(hyp)
    out: List[Tuple[Optional[str], int]] = []
    for w in words:
        best, best_ov, n_cover = None, 0.0, 0
        for lab in sorted(by):
            if w.end > w.start:
                ov = _overlap_with(by[lab], w.start, w.end)
            else:
                ov = 1.0 if _covering(by[lab], w.start) else 0.0
            n_cover += ov > 0
            if ov > best_ov:
                best, best_ov = lab, ov
        out.append((best, n_cover))
    return out


def assign(words: Sequence[Word], hyp: Sequence[Segment]) -> List[Optional[str]]:
    return [lab for lab, _ in assign_with_count(words, hyp)]


def mappings(ref: Sequence[Segment], hyp: Sequence[Segment], uem: Sequence[Region],
             step: float) -> Dict[str, Dict[str, str]]:
    """{'optimal': {hyp: ref}, 'majority_single': {hyp: ref}}"""
    opt = {h: r for r, h in frame_arrays(ref, hyp, uem, 0.0, False, step, "all").mapping.items()}
    ov = overlap_seconds(ref, hyp, uem, step, "single")
    maj: Dict[str, str] = {}
    for h in sorted({s.speaker for s in hyp}):
        cand = [(sec, r) for (r, hh), sec in ov.items() if hh == h and sec > 0]
        if cand:
            maj[h] = max(cand, key=lambda x: (x[0], x[1]))[1]
    return {"optimal": opt, "majority_single": maj}


def score_words(words: Sequence[Word], ref: Sequence[Segment], hyp: Sequence[Segment],
                uem: Sequence[Region], mapping: Dict[str, str]) -> Dict[str, Counter]:
    """Outcome counts per region ('all', 'single', 'overlap') for one mapping."""
    timeline = count_timeline(ref, uem)
    starts = [p[0] for p in timeline]
    ref_by = per_speaker_union(ref)
    labels = assign_with_count(words, hyp)
    out = {"all": Counter(), "single": Counter(), "overlap": Counter()}
    for w, (lab, n_cover) in zip(words, labels):
        mid = 0.5 * (w.start + w.end)
        i = bisect_right(starts, mid) - 1
        n_active = timeline[i][2] if i >= 0 and timeline[i][0] <= mid < timeline[i][1] else 0
        region = "overlap" if n_active >= 2 else "single"
        if lab is None:
            outcome = "unassigned"
        elif lab not in mapping:
            outcome = "unmapped"
        elif mapping[lab] == w.speaker:
            outcome = "correct"
        elif _covering(ref_by.get(mapping[lab], []), mid):
            outcome = "other_active_rule" if n_cover >= 2 else "other_active_missed"
        else:
            outcome = "inactive"
        out["all"][outcome] += 1
        out[region][outcome] += 1
    return out
