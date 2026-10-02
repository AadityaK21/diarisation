"""Run md-eval-22.pl on one file and parse its component times."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..config import REPO_ROOT
from .der import DERResult

MDEVAL = REPO_ROOT / "third_party" / "md-eval-22.pl"
_PAT = {
    "scored_speaker_s": r"SCORED SPEAKER TIME =\s*([\d.]+)",
    "missed_s": r"MISSED SPEAKER TIME =\s*([\d.]+)",
    "false_alarm_s": r"FALARM SPEAKER TIME =\s*([\d.]+)",
    "confusion_s": r"SPEAKER ERROR TIME =\s*([\d.]+)",
}


def run(ref: Path, hyp: Path, uem: Path, collar: float, exclude_overlap: bool) -> DERResult:
    cmd = ["perl", str(MDEVAL), "-r", str(ref), "-s", str(hyp), "-u", str(uem), "-c", str(collar)]
    if exclude_overlap:
        cmd.append("-1")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"md-eval failed ({p.returncode}): {p.stderr.strip()[-500:]}")
    vals = {}
    for k, pat in _PAT.items():
        m = re.search(pat, p.stdout)
        if not m:
            raise RuntimeError(f"md-eval output missing {k}:\n{p.stdout[-800:]}")
        vals[k] = float(m.group(1))
    return DERResult(off_grid=0, **vals)
