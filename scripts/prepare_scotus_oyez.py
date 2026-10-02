"""Build the raw-Oyez Supreme Court reference (transcript JSON only, no audio)."""

import argparse
import json

from diarisation.config import load_paths, load_yaml
from diarisation.data import scotus_oyez


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/scotus_oyez.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print(json.dumps(scotus_oyez.prepare(cfg, load_paths()["cache_root"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
