"""Overlap-conditioned DER decomposition.

Each scored frame is assigned to a region by the REFERENCE speaker count,
taken before collars are removed:

    nonspeech   n_ref == 0   (only false alarm can occur here)
    single      n_ref == 1
    overlap     n_ref >= 2

Collars and overlap exclusion then remove frames inside each region exactly
as for total DER, so the three regions partition the scored frames and their
components sum to the md-eval-validated total. With overlap excluded the
overlap region is empty by construction and is reported as such, not as 0%.
"""

from __future__ import annotations

from typing import Dict

from .der import DERResult, FrameArrays, components

REGIONS = ("nonspeech", "single", "overlap")


def decompose(fa: FrameArrays) -> Dict[str, DERResult]:
    masks = {
        "nonspeech": fa.n_ref == 0,
        "single": fa.n_ref == 1,
        "overlap": fa.n_ref >= 2,
    }
    out = {"all": components(fa)}
    for r in REGIONS:
        out[r] = components(fa, masks[r])
    return out


def partition_residual(parts: Dict[str, DERResult]) -> float:
    """Largest |total - sum of regions| over the four components (seconds)."""
    worst = 0.0
    for k in ("scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"):
        total = getattr(parts["all"], k)
        summed = sum(getattr(parts[r], k) for r in REGIONS)
        worst = max(worst, abs(total - summed))
    return worst
