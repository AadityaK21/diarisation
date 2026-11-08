"""Speaker-count confusion between a reference and a system output.

Both timelines partition the same UEM into pieces of constant active-speaker
count (intervals.count_timeline); sweeping them together gives the seconds
spent in every (reference count, system count) pair. Overlap detection
recall and precision follow directly: recall = T(ref >= 2, sys >= 2) /
T(ref >= 2), precision = T(ref >= 2, sys >= 2) / T(sys >= 2).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence, Tuple

from ..intervals import count_timeline
from ..io.rttm import Segment
from ..io.uem import Region


def count_confusion(ref: Sequence[Segment], hyp: Sequence[Segment],
                    uem: Sequence[Region]) -> Dict[Tuple[int, int], float]:
    """{(reference speaker count, system speaker count): seconds} inside the UEM."""
    a, b = count_timeline(ref, uem), count_timeline(hyp, uem)
    out: Dict[Tuple[int, int], float] = defaultdict(float)
    i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out[(a[i][2], b[j][2])] += e - s
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return dict(out)


def capped(conf: Dict[Tuple[int, int], float], cap: int = 3) -> Dict[Tuple[int, int], float]:
    """Merge counts >= cap into cap (e.g. 3 means '3 or more')."""
    out: Dict[Tuple[int, int], float] = defaultdict(float)
    for (r, h), sec in conf.items():
        out[(min(r, cap), min(h, cap))] += sec
    return dict(out)
