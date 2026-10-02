"""Exact interval arithmetic over reference speaker turns.

Everything here works on segment boundaries directly, with no frame grid, so
durations carry no quantisation error. The scorer (step 3) is checked against
these functions as well as against md-eval.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .io.rttm import Segment
from .io.uem import Region


def merge_regions(regions: Sequence[Region]) -> List[Region]:
    """Union of possibly overlapping or touching regions, sorted."""
    out: List[Region] = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def per_speaker_union(segments: Sequence[Segment]) -> Dict[str, List[Region]]:
    """One speaker overlapping themselves is one speaker, not overlap."""
    by_spk: Dict[str, List[Region]] = defaultdict(list)
    for s in segments:
        by_spk[s.speaker].append((s.start, s.end))
    return {spk: merge_regions(r) for spk, r in by_spk.items()}


def intersect(a: Sequence[Region], b: Sequence[Region]) -> List[Region]:
    """Intersection of two sorted, internally disjoint region lists."""
    out: List[Region] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def count_timeline(segments: Sequence[Segment], uem: Sequence[Region]) -> List[Tuple[float, float, int]]:
    """Partition the UEM into maximal pieces of constant active-speaker count.

    Returns [(start, end, n_active)], covering every UEM region exactly,
    including n_active == 0 (non-speech) pieces.
    """
    uem = merge_regions(uem)
    events: List[Tuple[float, int]] = []
    for regions in per_speaker_union(segments).values():
        for start, end in intersect(regions, uem):
            events.append((start, +1))
            events.append((end, -1))
    for start, end in uem:
        events.append((start, 0))
        events.append((end, 0))
    events.sort()

    uem_bounds = uem
    pieces: List[Tuple[float, float, int]] = []
    active = 0
    prev = None
    for t, delta in events:
        if prev is not None and t > prev:
            mid = 0.5 * (prev + t)
            if any(lo <= mid < hi for lo, hi in uem_bounds):
                if pieces and pieces[-1][2] == active and pieces[-1][1] == prev:
                    pieces[-1] = (pieces[-1][0], t, active)
                else:
                    pieces.append((prev, t, active))
        active += delta
        prev = t
    return pieces


def count_profile(segments: Sequence[Segment], uem: Sequence[Region]) -> Dict[int, float]:
    """Seconds of UEM time at each number of simultaneously active speakers."""
    prof: Dict[int, float] = defaultdict(float)
    for start, end, n in count_timeline(segments, uem):
        prof[n] += end - start
    return dict(prof)


def outside_uem(segments: Sequence[Segment], uem: Sequence[Region]) -> float:
    """Reference speaker-seconds that fall outside the UEM (should be ~0)."""
    uem = merge_regions(uem)
    total = 0.0
    for regions in per_speaker_union(segments).values():
        inside = sum(e - s for s, e in intersect(regions, uem))
        total += sum(e - s for s, e in regions) - inside
    return total


def subtract(a: Sequence[Region], b: Sequence[Region]) -> List[Region]:
    """a minus b, both sorted and internally disjoint."""
    out: List[Region] = []
    j = 0
    for lo, hi in a:
        cur = lo
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < hi:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < hi:
            out.append((cur, hi))
    return out
