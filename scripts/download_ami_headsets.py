"""Download the individual headset channels of AMI meetings (for the
per-word loudness analysis), into <audio_root>/amicorpus/<uri>/audio/.

Channel numbers come from AMI's meetings.xml (NXT manual annotations), so only
headsets worn by a speaker are fetched. Resumable, never inside the repository.

    python scripts/download_ami_headsets.py --partitions test
"""

import argparse
import sys
import xml.etree.ElementTree as ET

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami_words import ensure_archive

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from download_ami_audio import fetch  # noqa: E402


def headset_channels(meetings_xml, uris):
    """{uri: {global speaker id: channel}}"""
    out = {}
    for m in ET.parse(meetings_xml).getroot().iter("meeting"):
        if m.get("observation") in uris:
            out[m.get("observation")] = {s.get("global_name"): s.get("channel") for s in m.iter("speaker")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/ami.yaml")
    ap.add_argument("--words-config", default="configs/data/ami_words.yaml")
    ap.add_argument("--partitions", nargs="+", default=["test"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg, wcfg, paths = load_yaml(args.config), load_yaml(args.words_config), load_paths()
    root = paths["audio_root"] / "amicorpus"
    if REPO_ROOT in root.resolve().parents or root.resolve() == REPO_ROOT:
        sys.exit(f"audio_root {root} is inside the repository; refusing.")
    meetings_xml = ensure_archive(wcfg, paths["cache_root"]) / "corpusResources" / "meetings.xml"
    uris = []
    for part in args.partitions:
        uris += (REPO_ROOT / cfg["out_dir"] / "lists" / f"{part}.txt").read_text().split()
    chans = headset_channels(meetings_xml, set(uris))
    failed = []
    for uri in uris:
        for spk, ch in sorted(chans[uri].items()):
            stream = f"Headset-{ch}"
            url = cfg["audio_url"].format(uri=uri, stream=stream)
            dest = root / uri / "audio" / url.rsplit("/", 1)[1]
            if args.dry_run:
                print(spk, url, "->", dest)
                continue
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                n = fetch(url, dest)
                print(f"{uri} {stream} ({spk}): {n / 1e6:.1f} MB", flush=True)
            except Exception as e:  # keep going, report at the end
                failed.append((uri, stream, repr(e)))
                print(f"FAILED {uri} {stream}: {e!r}", flush=True)
    if failed:
        print(f"{len(failed)} downloads failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
