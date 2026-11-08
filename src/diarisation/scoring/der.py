"""Frame-level DER with md-eval semantics.

The timeline is cut into frames of `frame_step` seconds. With RTTM times at
millisecond precision and a 1 ms step, every speaker boundary falls exactly on
a frame edge, so frame counting is exact rather than an approximation; speaker
boundaries that are not on the grid are counted and reported, never silently
rounded away. UEM boundaries need not be on the grid (AMI UEMs end at sample
precision): a frame cut by a UEM boundary is weighted by the fraction of it
inside the UEM, which keeps the result exact.

Semantics follow md-eval-22.pl (see third_party/README.md):

  * mapping: one-to-one ref->hyp assignment maximising overlap time, computed
    on the evaluation UEM before collars or overlap exclusion are applied;
  * collar: no-score zone of +/- `collar` s around every reference segment
    start and end;
  * exclude_overlap (md-eval -1): drop frames where >= 2 reference speakers
    are active;
  * per scored frame with n_ref reference speakers, n_hyp system speakers and
    n_correct mapped matches:
        missed    = max(n_ref - n_hyp, 0)
        false     = max(n_hyp - n_ref, 0)
        confusion = min(n_ref, n_hyp) - n_correct
    each weighted by frame duration, all over scored reference speaker time.

`frame_arrays` exposes the per-frame quantities so the overlap-conditioned
decomposition (step 4) is computed from the same arrays as total DER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..intervals import merge_regions
from ..io.rttm import Segment
from ..io.uem import Region

GRID_TOL = 1e-6  # fraction of a frame; tolerates float noise in parsed times


class SelfOverlapError(ValueError):
    """A speaker overlaps themselves; md-eval treats this as fatal too."""


@dataclass
class FrameArrays:
    step: float
    n_ref: np.ndarray        # int16 per frame
    n_hyp: np.ndarray
    n_correct: np.ndarray
    eval_mask: np.ndarray    # any part of the frame inside UEM
    score_mask: np.ndarray   # eval minus collars minus (optionally) overlap
    weight: np.ndarray       # fraction of each frame inside UEM (1.0 almost everywhere)
    mapping: Dict[str, str]  # ref -> hyp
    off_grid: int            # boundaries not on the frame grid


@dataclass
class DERResult:
    scored_speaker_s: float
    missed_s: float
    false_alarm_s: float
    confusion_s: float
    off_grid: int

    @property
    def der(self) -> float:
        e = self.missed_s + self.false_alarm_s + self.confusion_s
        return e / self.scored_speaker_s if self.scored_speaker_s else float("nan")


def _check_self_overlap(segs: Sequence[Segment], which: str) -> None:
    by: Dict[str, List[Segment]] = {}
    for s in segs:
        by.setdefault(s.speaker, []).append(s)
    for spk, ss in by.items():
        ss.sort(key=lambda s: s.start)
        for a, b in zip(ss, ss[1:]):
            if b.start < a.end - 1e-8:
                raise SelfOverlapError(
                    f"{which}: speaker {spk} overlaps themselves at "
                    f"({a.start}, {a.end}) / ({b.start}, {b.end})")


class _Grid:
    def __init__(self, step: float, n: int):
        self.step, self.n, self.off_grid = step, n, 0

    def idx(self, t: float) -> int:
        x = t / self.step
        k = int(round(x))
        if abs(x - k) > GRID_TOL:
            self.off_grid += 1
        return min(max(k, 0), self.n)


def _activity(segs: Sequence[Segment], grid: _Grid) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for s in segs:
        if s.end <= s.start:
            continue
        a = out.setdefault(s.speaker, np.zeros(grid.n, dtype=bool))
        a[grid.idx(s.start):grid.idx(s.end)] = True
    return out


def _uem_weight(uem: Sequence[Region], step: float, n: int) -> np.ndarray:
    """Fraction of each frame inside the UEM, exact for off-grid boundaries."""
    w = np.zeros(n, dtype=np.float64)
    for lo, hi in merge_regions(uem):
        x0, x1 = lo / step, hi / step
        k0, k1 = int(np.floor(x0 + GRID_TOL)), int(np.floor(x1 + GRID_TOL))
        if k0 == k1:
            w[k0] += x1 - x0
            continue
        w[k0] += (k0 + 1) - x0
        w[k0 + 1:k1] += 1.0
        if k1 < n:
            w[k1] += x1 - k1
    np.clip(w, 0.0, 1.0, out=w)
    w[np.abs(w) < GRID_TOL] = 0.0
    w[np.abs(w - 1.0) < GRID_TOL] = 1.0
    return w


def _optimal_mapping(ref: Dict[str, np.ndarray], hyp: Dict[str, np.ndarray],
                     weight: np.ndarray) -> Dict[str, str]:
    if not ref or not hyp:
        return {}
    r_names, h_names = sorted(ref), sorted(hyp)
    overlap = np.zeros((len(r_names), len(h_names)), dtype=np.float64)
    for i, r in enumerate(r_names):
        rw = np.where(ref[r], weight, 0.0)
        for j, h in enumerate(h_names):
            overlap[i, j] = rw[hyp[h]].sum()
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    # Pairs with zero overlap contribute nothing; md-eval leaves them unmapped.
    return {r_names[i]: h_names[j] for i, j in zip(rows, cols) if overlap[i, j] > 0}


def frame_arrays(ref_segs: Sequence[Segment], hyp_segs: Sequence[Segment], uem: Sequence[Region],
                 collar: float, exclude_overlap: bool, step: float,
                 mapping_region: str = "all") -> FrameArrays:
    """`mapping_region` = "all" (md-eval: every evaluated frame) or "single"
    (only frames with exactly one reference speaker: the mapping is fixed on
    unambiguous speech and then applied unchanged everywhere, overlap included)."""
    _check_self_overlap(ref_segs, "reference")
    _check_self_overlap(hyp_segs, "hypothesis")

    end = max([e for _, e in uem] + [s.end for s in ref_segs] + [s.end for s in hyp_segs] + [0.0])
    grid = _Grid(step, int(np.ceil(end / step - GRID_TOL)) + 1)

    weight = _uem_weight(uem, step, grid.n)
    eval_mask = weight > 0

    ref = _activity(ref_segs, grid)
    hyp = _activity(hyp_segs, grid)
    zeros = np.zeros(grid.n, dtype=np.int16)
    n_ref = sum((a.astype(np.int16) for a in ref.values()), zeros.copy())
    n_hyp = sum((a.astype(np.int16) for a in hyp.values()), zeros.copy())

    if mapping_region == "all":
        map_weight = weight
    elif mapping_region == "single":
        map_weight = np.where(n_ref == 1, weight, 0.0)
    else:
        raise ValueError(f"unknown mapping_region {mapping_region!r}")
    mapping = _optimal_mapping(ref, hyp, map_weight)
    n_correct = zeros.copy()
    for r, h in mapping.items():
        n_correct += (ref[r] & hyp[h]).astype(np.int16)

    score_mask = eval_mask.copy()
    if collar > 0:
        for s in ref_segs:
            if s.end <= s.start:
                continue
            for b in (s.start, s.end):
                score_mask[grid.idx(b - collar):grid.idx(b + collar)] = False
    if exclude_overlap:
        score_mask &= n_ref < 2

    return FrameArrays(step, n_ref, n_hyp, n_correct, eval_mask, score_mask, weight, mapping, grid.off_grid)


def components(fa: FrameArrays, mask: np.ndarray | None = None) -> DERResult:
    """Sum the DER components over `mask` (default: the score mask)."""
    m = fa.score_mask if mask is None else (fa.score_mask & mask)
    w = fa.weight[m] * fa.step
    nr = fa.n_ref[m].astype(np.int64)
    nh = fa.n_hyp[m].astype(np.int64)
    nc = fa.n_correct[m].astype(np.int64)
    return DERResult(
        scored_speaker_s=float((nr * w).sum()),
        missed_s=float((np.maximum(nr - nh, 0) * w).sum()),
        false_alarm_s=float((np.maximum(nh - nr, 0) * w).sum()),
        confusion_s=float(((np.minimum(nr, nh) - nc) * w).sum()),
        off_grid=fa.off_grid,
    )


def score(ref_segs, hyp_segs, uem, collar: float, exclude_overlap: bool, step: float) -> DERResult:
    return components(frame_arrays(ref_segs, hyp_segs, uem, collar, exclude_overlap, step))


def overlap_seconds(ref_segs: Sequence[Segment], hyp_segs: Sequence[Segment], uem: Sequence[Region],
                    step: float, region: str = "all") -> Dict[Tuple[str, str], float]:
    """Seconds each (ref, hyp) pair is active together inside the UEM.

    region "all" = every evaluated frame; "single" = frames with exactly one
    reference speaker. Same grid and UEM weighting as `frame_arrays`.
    """
    end = max([e for _, e in uem] + [s.end for s in ref_segs] + [s.end for s in hyp_segs] + [0.0])
    grid = _Grid(step, int(np.ceil(end / step - GRID_TOL)) + 1)
    weight = _uem_weight(uem, step, grid.n)
    ref, hyp = _activity(ref_segs, grid), _activity(hyp_segs, grid)
    if region == "single":
        n_ref = sum((a.astype(np.int16) for a in ref.values()), np.zeros(grid.n, dtype=np.int16))
        weight = np.where(n_ref == 1, weight, 0.0)
    elif region != "all":
        raise ValueError(f"unknown region {region!r}")
    out = {}
    for r, ra in ref.items():
        rw = np.where(ra, weight, 0.0)
        for h, ha in hyp.items():
            out[(r, h)] = float(rw[ha].sum()) * step
    return out
