"""Speaker-attributed WER: cpWER (permutation-invariant) and fixed-mapping WER,
with every error assigned to the reference region it falls in.

Words are normalised one by one with Whisper's EnglishTextNormalizer (fillers
removed, contractions expanded); a word can become zero or several tokens,
each keeping the word's timing. Per-word normalisation does not join number
words ("twenty five" -> "20 5" vs "25"); the effect is the same for every
system compared here and is noted in docs/decisions.md.

Errors under a stream assignment are counted with Levenshtein edit operations
per matched (reference speaker, system stream) pair:
  substitution, deletion -> region of the reference token
  insertion              -> region of the system token (reference speaker
                            count at its midpoint)
Unmatched reference streams are all deletions; unmatched system streams are
all insertions.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from ..intervals import count_timeline
from ..io.rttm import Segment
from ..io.uem import Region

_NORM = None


def normaliser():
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _NORM = EnglishTextNormalizer()
    return _NORM


@dataclass(frozen=True)
class Token:
    text: str
    start: float
    end: float


def tokens(words: Sequence[Tuple[str, float, float]]) -> List[Token]:
    """[(word, start, end)] -> normalised tokens, time-ordered."""
    n = normaliser()
    out = []
    for w, s, e in sorted(words, key=lambda x: (x[1], x[2])):
        for t in n(w).split():
            out.append(Token(t, s, e))
    return out


class RegionLookup:
    def __init__(self, ref: Sequence[Segment], uem: Sequence[Region]):
        self.tl = count_timeline(ref, uem)
        self.starts = [p[0] for p in self.tl]

    def __call__(self, tok: Token) -> str:
        mid = 0.5 * (tok.start + tok.end)
        i = bisect_right(self.starts, mid) - 1
        n = self.tl[i][2] if i >= 0 and self.tl[i][0] <= mid < self.tl[i][1] else 0
        return "overlap" if n >= 2 else "single"


def _count_pair(ref: List[Token], hyp: List[Token], region, acc: Dict[str, Dict[str, int]]) -> None:
    for op in Levenshtein.editops([t.text for t in ref], [t.text for t in hyp]):
        if op.tag == "replace":
            acc[region(ref[op.src_pos])]["sub"] += 1
        elif op.tag == "delete":
            acc[region(ref[op.src_pos])]["del"] += 1
        else:
            acc[region(hyp[op.dest_pos])]["ins"] += 1


def _empty_acc():
    return {r: {"n_ref": 0, "sub": 0, "del": 0, "ins": 0} for r in ("single", "overlap")}


def score_assignment(ref_streams: Dict[str, List[Token]], hyp_streams: Dict[str, List[Token]],
                     pairs: Dict[str, Optional[str]], region) -> Dict[str, Dict[str, int]]:
    """pairs: ref speaker -> hyp stream key (or None). Each hyp key used at most once."""
    acc = _empty_acc()
    for toks in ref_streams.values():
        for t in toks:
            acc[region(t)]["n_ref"] += 1
    used = set()
    for r, toks in ref_streams.items():
        h = pairs.get(r)
        _count_pair(toks, hyp_streams.get(h, []) if h is not None else [], region, acc)
        if h is not None:
            used.add(h)
    for h, toks in hyp_streams.items():
        if h not in used:
            for t in toks:
                acc[region(t)]["ins"] += 1
    return acc


def cp_assignment(ref_streams: Dict[str, List[Token]], hyp_streams: Dict[str, List[Token]]) -> Dict[str, Optional[str]]:
    """The cpWER permutation: minimum total edit distance, one-to-one, with
    unmatched streams costing all their tokens."""
    R, H = sorted(ref_streams), sorted(hyp_streams)
    n = max(len(R), len(H))
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = [t.text for t in ref_streams[R[i]]] if i < len(R) else []
            h = [t.text for t in hyp_streams[H[j]]] if j < len(H) else []
            cost[i, j] = Levenshtein.distance(r, h)
    rows, cols = linear_sum_assignment(cost)
    return {R[i]: (H[j] if j < len(H) else None) for i, j in zip(rows, cols) if i < len(R)}


def merge_by_mapping(hyp_streams: Dict[str, List[Token]], mapping: Dict[str, str]) -> Dict[str, List[Token]]:
    """Many-to-one fixed mapping: system streams mapped to the same reference
    speaker are merged in time order; unmapped system streams stay separate."""
    out: Dict[str, List[Token]] = defaultdict(list)
    for h, toks in hyp_streams.items():
        out[f"ref:{mapping[h]}" if h in mapping else f"unmapped:{h}"].extend(toks)
    return {k: sorted(v, key=lambda t: (t.start, t.end)) for k, v in out.items()}
