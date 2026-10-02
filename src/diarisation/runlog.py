"""Run provenance: config hash, git commit, package versions, seed, wall time."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import REPO_ROOT, config_hash


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_state() -> Dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    # Results are only reproducible from a commit if the code was committed.
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "configs", "Makefile")
    return {"commit": commit, "code_dirty": bool(dirty) and dirty != "unknown"}


def versions(packages: Iterable[str]) -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


class RunLog:
    def __init__(self, experiment: str, config: Dict[str, Any], seed: Optional[int],
                 packages: Iterable[str] = ("numpy", "PyYAML")):
        self.record: Dict[str, Any] = {
            "experiment": experiment,
            "config_hash": config_hash(config),
            "config": config,
            "seed": seed,
            "git": git_state(),
            "versions": versions(packages),
            "argv": sys.argv,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._t0 = time.perf_counter()

    @property
    def short_hash(self) -> str:
        return self.record["config_hash"][:10]

    def finish(self, out_dir: Path, **extra: Any) -> Path:
        self.record["wall_seconds"] = round(time.perf_counter() - self._t0, 3)
        self.record.update(extra)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "run.json"
        path.write_text(json.dumps(self.record, indent=2, default=str) + "\n", encoding="utf-8")
        return path
