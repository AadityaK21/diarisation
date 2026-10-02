"""AMI reference preparation from the BUT/pyannote diarisation setup.

The upstream repo is cloned into the cache at a pinned commit, and its RTTM,
UEM and partition lists are copied byte-for-byte into data/ami/. Copying
rather than re-serialising keeps the committed references identical to the
published ones, which the manifest's checksums let anyone verify.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..config import REPO_ROOT
from ..intervals import outside_uem
from ..io.rttm import read_rttm
from ..io.uem import read_uem


def ensure_upstream(cfg: Dict[str, Any], cache_root: Path) -> Path:
    dest = cache_root / "AMI-diarization-setup"
    want = cfg["source"]["commit"]
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", cfg["source"]["repo"], str(dest)], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != want:
        subprocess.run(["git", "fetch", "--quiet", "origin", want], cwd=dest, check=True)
        subprocess.run(["git", "checkout", "--quiet", want], cwd=dest, check=True)
    return dest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meetings(upstream: Path, partition: str) -> List[str]:
    lst = upstream / "lists" / f"{partition}.meetings.txt"
    return [l.strip() for l in lst.read_text().splitlines() if l.strip()]


def prepare(cfg: Dict[str, Any], upstream: Path) -> Dict[str, Any]:
    out = REPO_ROOT / cfg["out_dir"]
    files: Dict[str, str] = {}
    checks: Dict[str, Any] = {}

    for part in cfg["partitions"]:
        uris = _meetings(upstream, part)
        expected = cfg["expected_meetings"][part]
        if len(uris) != expected:
            raise RuntimeError(f"{part}: {len(uris)} meetings listed, expected {expected}")
        if len(set(uris)) != len(uris):
            raise RuntimeError(f"{part}: duplicate meetings in list")

        dst = out / "lists" / f"{part}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / "lists" / f"{part}.meetings.txt", dst)
        files[str(dst.relative_to(REPO_ROOT))] = _sha256(dst)

        for uri in uris:
            src = upstream / "uems" / part / f"{uri}.uem"
            dst = out / "uem" / part / f"{uri}.uem"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            files[str(dst.relative_to(REPO_ROOT))] = _sha256(dst)

        for ann in cfg["annotations"]:
            n_seg = 0
            outside = 0.0
            for uri in uris:
                src = upstream / ann / "rttms" / part / f"{uri}.rttm"
                dst = out / ann / "rttm" / part / f"{uri}.rttm"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                files[str(dst.relative_to(REPO_ROOT))] = _sha256(dst)

                ref = read_rttm(dst)
                if set(ref) != {uri}:
                    raise RuntimeError(f"{dst}: RTTM uris {sorted(ref)} != {uri}")
                uem = read_uem(out / "uem" / part / f"{uri}.uem")
                if set(uem) != {uri}:
                    raise RuntimeError(f"UEM for {uri} names {sorted(uem)}")
                n_seg += len(ref[uri])
                outside += outside_uem(ref[uri], uem[uri])
            checks[f"{ann}/{part}"] = {
                "meetings": len(uris),
                "segments": n_seg,
                "speaker_seconds_outside_uem": round(outside, 6),
            }

    manifest = {
        "source": cfg["source"],
        "annotations": cfg["annotations"],
        "checks": checks,
        "sha256": dict(sorted(files.items())),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audio_urls(cfg: Dict[str, Any], condition: str, partitions: List[str]) -> List[tuple[str, str]]:
    """[(uri, url)] for one microphone condition, read from the committed lists."""
    stream = cfg["conditions"][condition]
    out = []
    for part in partitions:
        lst = REPO_ROOT / cfg["out_dir"] / "lists" / f"{part}.txt"
        for uri in lst.read_text().split():
            out.append((uri, cfg["audio_url"].format(uri=uri, stream=stream)))
    return out
