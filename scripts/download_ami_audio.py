"""Download AMI audio for one microphone condition into <audio_root>/amicorpus/.

Resumable: complete files are skipped, partial downloads are kept as .part and
continued. Audio is never written inside the repository.

    python scripts/download_ami_audio.py --condition ihm_mix --partitions test
    python scripts/download_ami_audio.py --condition sdm --partitions dev test
    python scripts/download_ami_audio.py --condition sdm --dry-run
"""

import argparse
import sys
from pathlib import Path

import requests

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami import audio_urls


def fetch(url: str, dest: Path) -> int:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # nothing left to fetch
            part.replace(dest)
            return 0
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        n = 0
        with open(part, mode) as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                n += len(chunk)
    part.replace(dest)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/ami.yaml")
    ap.add_argument("--condition", required=True, choices=["ihm_mix", "sdm"])
    ap.add_argument("--partitions", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--dry-run", action="store_true", help="print URLs and exit")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = load_paths()["audio_root"] / "amicorpus"
    if REPO_ROOT in root.resolve().parents or root.resolve() == REPO_ROOT:
        sys.exit(f"audio_root {root} is inside the repository; refusing.")

    jobs = audio_urls(cfg, args.condition, args.partitions)
    failed = []
    for i, (uri, url) in enumerate(jobs, 1):
        dest = root / uri / "audio" / url.rsplit("/", 1)[1]
        if args.dry_run:
            print(url, "->", dest)
            continue
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            n = fetch(url, dest)
            print(f"[{i}/{len(jobs)}] {uri} {n / 1e6:.1f} MB")
        except requests.RequestException as exc:
            print(f"[{i}/{len(jobs)}] {uri} FAILED: {exc}", file=sys.stderr)
            failed.append(uri)
    if failed:
        print(f"{len(failed)} downloads failed: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
