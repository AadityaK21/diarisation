"""Config loading. All tunables live in YAML under configs/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_paths() -> Dict[str, Path]:
    """Per-machine paths. Falls back to the example file, loudly."""
    local = REPO_ROOT / "configs" / "paths.yaml"
    src = local if local.exists() else REPO_ROOT / "configs" / "paths.example.yaml"
    if src != local:
        print(f"[paths] configs/paths.yaml not found; using defaults from {src.name}")
    raw = load_yaml(src)
    return {k: Path(str(v)).expanduser() for k, v in raw.items()}


def config_hash(obj: Any) -> str:
    """SHA-256 over canonical JSON. Key order and whitespace do not matter."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
