"""RTTM reading and writing. Only SPEAKER records are used."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: str | Path) -> Dict[str, List[Segment]]:
    """Return {uri: [Segment, ...]} sorted by start time.

    Raises on malformed lines rather than skipping them: a silently dropped
    reference turn changes every number computed from the file.
    """
    out: Dict[str, List[Segment]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            fields = line.split()
            if not fields or fields[0].startswith(";"):
                continue
            if len(fields) < 8:
                raise ValueError(f"{path}:{lineno}: expected >= 8 fields, got {len(fields)}")
            if fields[0] != "SPEAKER":
                continue
            uri, onset, dur, spk = fields[1], float(fields[3]), float(fields[4]), fields[7]
            if dur < 0:
                raise ValueError(f"{path}:{lineno}: negative duration {dur}")
            out[uri].append(Segment(onset, onset + dur, spk))
    for segs in out.values():
        segs.sort(key=lambda s: (s.start, s.end, s.speaker))
    return dict(out)


def write_rttm(path: str | Path, uri: str, segments: Iterable[Segment], channel: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in sorted(segments, key=lambda s: (s.start, s.end, s.speaker)):
            if " " in s.speaker:
                raise ValueError(f"RTTM speaker label may not contain spaces: {s.speaker!r}")
            fh.write(
                f"SPEAKER {uri} {channel} {s.start:.3f} {s.end - s.start:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>\n"
            )
