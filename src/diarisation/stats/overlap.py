"""Per-file reference overlap statistics and their aggregation.

Two overlap measures, because they answer different questions:

  overlap_time_frac     share of speech TIME with >= 2 active speakers.
                        This is what "AMI has X% overlap" usually means.
  overlap_spk_time_frac share of reference SPEAKER-time inside overlap.
                        DER's denominator is speaker-time, so this is the
                        share of what DER scores that lies in overlap. It is
                        always larger than the first.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..intervals import count_profile
from ..io.rttm import Segment
from ..io.uem import Region
from . import bootstrap

PER_FILE_FIELDS = [
    "dataset", "annotation", "partition", "uri", "n_speakers",
    "uem_s", "nonspeech_s", "speech_s", "overlap_s", "overlap3plus_s",
    "speaker_time_s", "overlap_speaker_time_s",
    "overlap_time_frac", "overlap_spk_time_frac",
]


def file_row(segments: Sequence[Segment], uem: Sequence[Region]) -> Dict[str, Any]:
    prof = count_profile(segments, uem)
    speech = sum(t for n, t in prof.items() if n >= 1)
    overlap = sum(t for n, t in prof.items() if n >= 2)
    spk_time = sum(n * t for n, t in prof.items())
    ovl_spk_time = sum(n * t for n, t in prof.items() if n >= 2)
    return {
        "n_speakers": len({s.speaker for s in segments}),
        "uem_s": sum(prof.values()),
        "nonspeech_s": prof.get(0, 0.0),
        "speech_s": speech,
        "overlap_s": overlap,
        "overlap3plus_s": sum(t for n, t in prof.items() if n >= 3),
        "speaker_time_s": spk_time,
        "overlap_speaker_time_s": ovl_spk_time,
        "overlap_time_frac": overlap / speech if speech else 0.0,
        "overlap_spk_time_frac": ovl_spk_time / spk_time if spk_time else 0.0,
    }


def write_per_file(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_FILE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


AGG_FIELDS = [
    "dataset", "annotation", "partition", "n_files", "hours_uem", "hours_speech",
    "overlap_time_frac", "overlap_time_frac_lo", "overlap_time_frac_hi",
    "overlap_spk_time_frac", "overlap_spk_time_frac_lo", "overlap_spk_time_frac_hi",
    "mean_file_overlap_time_frac", "mean_file_lo", "mean_file_hi",
    "min_file_overlap_time_frac", "max_file_overlap_time_frac",
    "n_resamples", "ci_level", "seed",
]


def aggregate_from_csv(per_file: Path, n_resamples: int, ci_level: float, seed: int) -> List[Dict[str, Any]]:
    """Aggregate strictly from the per-file CSV, never from in-memory values."""
    with open(per_file, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    groups: Dict[tuple, List[Dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["annotation"], r["partition"]), []).append(r)

    out = []
    for (ds, ann, part), rs in groups.items():
        f = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        pooled = bootstrap.ratio_of_sums(f("overlap_s"), f("speech_s"), n_resamples, ci_level, seed)
        pooled_spk = bootstrap.ratio_of_sums(f("overlap_speaker_time_s"), f("speaker_time_s"),
                                             n_resamples, ci_level, seed)
        per_file_frac = f("overlap_time_frac")
        mean = bootstrap.mean(per_file_frac, n_resamples, ci_level, seed)
        out.append({
            "dataset": ds, "annotation": ann, "partition": part, "n_files": len(rs),
            "hours_uem": sum(f("uem_s")) / 3600, "hours_speech": sum(f("speech_s")) / 3600,
            "overlap_time_frac": pooled.point, "overlap_time_frac_lo": pooled.lo,
            "overlap_time_frac_hi": pooled.hi,
            "overlap_spk_time_frac": pooled_spk.point, "overlap_spk_time_frac_lo": pooled_spk.lo,
            "overlap_spk_time_frac_hi": pooled_spk.hi,
            "mean_file_overlap_time_frac": mean.point, "mean_file_lo": mean.lo, "mean_file_hi": mean.hi,
            "min_file_overlap_time_frac": min(per_file_frac),
            "max_file_overlap_time_frac": max(per_file_frac),
            "n_resamples": n_resamples, "ci_level": ci_level, "seed": seed,
        })
    return out


def write_aggregate(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AGG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
