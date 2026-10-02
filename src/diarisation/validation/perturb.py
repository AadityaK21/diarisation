"""Synthetic diarisation systems: seeded corruptions of a reference.

Used only to validate the scorer. Each corruption targets one DER component
so a semantic mismatch with md-eval shows up in a specific column.
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Sequence

import numpy as np

from ..intervals import merge_regions, per_speaker_union, subtract
from ..io.rttm import Segment
from ..io.uem import Region


def rng_for(seed: int, uri: str, system: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([seed, zlib.crc32(f"{uri}/{system}".encode())]))


def _speakers(segs: Sequence[Segment]) -> List[str]:
    return sorted({s.speaker for s in segs})


def jitter(segs, rng, boundary_sd, **_):
    out = []
    for s in segs:
        a, b = s.start + rng.normal(0, boundary_sd), s.end + rng.normal(0, boundary_sd)
        if b > a:
            out.append(Segment(max(a, 0.0), b, s.speaker))
    return out


def swap(segs, rng, frac, **_):
    spk = _speakers(segs)
    out = []
    for s in segs:
        if len(spk) > 1 and rng.random() < frac:
            others = [x for x in spk if x != s.speaker]
            out.append(Segment(s.start, s.end, others[rng.integers(len(others))]))
        else:
            out.append(s)
    return out


def miss_fa(segs, rng, uem, drop_frac, add_frac, add_min, add_max, **_):
    spk = _speakers(segs) + ["FA_extra"]
    out = [s for s in segs if rng.random() >= drop_frac]
    lo, hi = uem[0][0], uem[-1][1]
    for _ in range(int(round(add_frac * len(segs)))):
        a = rng.uniform(lo, hi)
        out.append(Segment(a, min(a + rng.uniform(add_min, add_max), hi), spk[rng.integers(len(spk))]))
    return out


def merge_split(segs, uem, **_):
    spk = _speakers(segs)
    mid = 0.5 * (uem[0][0] + uem[-1][1])
    out = []
    for s in segs:
        lab = s.speaker
        if len(spk) >= 2 and lab == spk[1]:
            lab = spk[0]
        elif len(spk) >= 3 and lab == spk[2] and s.start >= mid:
            lab = spk[2] + "_b"
        out.append(Segment(s.start, s.end, lab))
    return out


def exclusive(segs, **_):
    """Overlap goes to the speaker with the most total speech; others lose it."""
    regions = per_speaker_union(segs)
    order = sorted(regions, key=lambda k: -sum(e - s for s, e in regions[k]))
    taken: List[Region] = []
    out = []
    for spk in order:
        for a, b in subtract(regions[spk], taken):
            out.append(Segment(a, b, spk))
        taken = merge_regions(taken + regions[spk])
    return out


def finalise(segs, rng) -> List[Segment]:
    """Anonymise labels, enforce no self-overlap, snap to the ms grid."""
    spk = _speakers(segs)
    names = [f"sys{i:02d}" for i in rng.permutation(len(spk))]
    rename = dict(zip(spk, names))
    out = []
    for lab, regions in per_speaker_union([Segment(s.start, s.end, rename[s.speaker]) for s in segs]).items():
        for a, b in regions:
            a, b = round(a, 3), round(b, 3)
            if b > a:
                out.append(Segment(a, b, lab))
    return sorted(out, key=lambda s: (s.start, s.speaker))


def make(system: str, params: Dict[str, Any], ref: Sequence[Segment], uem: Sequence[Region],
         seed: int, uri: str) -> List[Segment]:
    rng = rng_for(seed, uri, system)
    segs = list(ref)
    if system == "relabel":
        pass
    elif system == "jitter":
        segs = jitter(segs, rng, **params)
    elif system == "swap":
        segs = swap(segs, rng, **params)
    elif system == "miss_fa":
        segs = miss_fa(segs, rng, uem, **params)
    elif system == "merge_split":
        segs = merge_split(segs, uem)
    elif system == "exclusive":
        segs = exclusive(segs)
    elif system == "combined":
        segs = jitter(segs, rng, **params)
        segs = swap(segs, rng, **params)
        segs = miss_fa(segs, rng, uem, **params)
    else:
        raise ValueError(f"unknown system {system}")
    return finalise(segs, rng)
