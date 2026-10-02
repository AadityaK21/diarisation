"""Percentile bootstrap over files (meetings).

The resampling unit is the file because files, not frames, are the
independent units: frames within a meeting share speakers, room and
recording chain, and resampling them would give intervals that are far too
narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n_files: int
    n_resamples: int
    ci_level: float
    seed: int


def _indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n, size=(n_resamples, n))


def _interval(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def ratio_of_sums(
    num: Sequence[float], den: Sequence[float], n_resamples: int, ci_level: float, seed: int
) -> Estimate:
    """Pooled ratio sum(num)/sum(den), e.g. total overlap time / total speech.

    This is how DER is aggregated over a corpus (time-weighted), so it is the
    primary aggregate. Long meetings weigh more.
    """
    num_a, den_a = np.asarray(num, float), np.asarray(den, float)
    if len(num_a) != len(den_a) or len(num_a) == 0:
        raise ValueError("num and den must be non-empty and the same length")
    point = float(num_a.sum() / den_a.sum())
    if len(num_a) == 1:
        return Estimate(point, float("nan"), float("nan"), 1, 0, ci_level, seed)
    idx = _indices(len(num_a), n_resamples, seed)
    samples = num_a[idx].sum(axis=1) / den_a[idx].sum(axis=1)
    lo, hi = _interval(samples, ci_level)
    return Estimate(point, lo, hi, len(num_a), n_resamples, ci_level, seed)


def mean(x: Sequence[float], n_resamples: int, ci_level: float, seed: int) -> Estimate:
    """Unweighted mean over files: every meeting counts once."""
    x_a = np.asarray(x, float)
    if len(x_a) == 0:
        raise ValueError("x must be non-empty")
    point = float(x_a.mean())
    if len(x_a) == 1:
        return Estimate(point, float("nan"), float("nan"), 1, 0, ci_level, seed)
    idx = _indices(len(x_a), n_resamples, seed)
    lo, hi = _interval(x_a[idx].mean(axis=1), ci_level)
    return Estimate(point, lo, hi, len(x_a), n_resamples, ci_level, seed)


def of_sums(columns: dict, fn, n_resamples: int, ci_level: float, seed: int) -> Estimate:
    """Any statistic that is a function of per-file column sums.

    `fn` receives {name: sum} and must work on numpy arrays as well as floats,
    so all resamples are evaluated at once. Every column is resampled with the
    SAME meeting indices, which makes differences between statistics (e.g.
    DER under two conventions) properly paired.
    """
    cols = {k: np.asarray(v, float) for k, v in columns.items()}
    n = len(next(iter(cols.values())))
    if n == 0 or any(len(v) != n for v in cols.values()):
        raise ValueError("columns must be non-empty and equal length")
    with np.errstate(invalid="ignore", divide="ignore"):
        point = float(fn({k: v.sum() for k, v in cols.items()}))
        if n == 1:
            return Estimate(point, float("nan"), float("nan"), 1, 0, ci_level, seed)
        idx = _indices(n, n_resamples, seed)
        samples = np.asarray(fn({k: v[idx].sum(axis=1) for k, v in cols.items()}), float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return Estimate(point, float("nan"), float("nan"), n, n_resamples, ci_level, seed)
    lo, hi = _interval(samples, ci_level)
    return Estimate(point, lo, hi, n, n_resamples, ci_level, seed)
