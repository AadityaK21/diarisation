"""Copy the pinned AMI diarisation references into data/ami/ and validate them."""

import argparse
import json

from diarisation.config import load_paths, load_yaml
from diarisation.data import ami


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/ami.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    upstream = ami.ensure_upstream(cfg, load_paths()["cache_root"])
    manifest = ami.prepare(cfg, upstream)
    print(json.dumps(manifest["checks"], indent=2))
    print(f"{len(manifest['sha256'])} files written under {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
