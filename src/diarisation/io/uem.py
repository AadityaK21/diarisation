"""UEM reading and writing: `<uri> <channel> <start> <end>` per line."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Region = Tuple[float, float]


def read_uem(path: str | Path) -> Dict[str, List[Region]]:
    out: Dict[str, List[Region]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            fields = line.split()
            if not fields or fields[0].startswith(";"):
                continue
            if len(fields) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 fields, got {len(fields)}")
            start, end = float(fields[2]), float(fields[3])
            if end < start:
                raise ValueError(f"{path}:{lineno}: end {end} < start {start}")
            out[fields[0]].append((start, end))
    for regions in out.values():
        regions.sort()
    return dict(out)


def write_uem(path: str | Path, uri: str, regions: Iterable[Region], channel: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for start, end in sorted(regions):
            fh.write(f"{uri} {channel} {start:.3f} {end:.3f}\n")
