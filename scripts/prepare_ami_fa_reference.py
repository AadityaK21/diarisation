"""Copy the forced-alignment AMI reference (configs/data/ami_fa.yaml) into
data/ami/forced_alignment/, renaming speakers <uri>.<letter> to AMI global
participant ids so it lines up with the only_words reference and the words.

Checks: the clone is at the pinned commit; every meeting of the partition is
present; every speaker letter maps to a participant of that meeting; the set
of speakers equals the only_words reference's. A summary goes to MANIFEST.json."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami_words import agent_map, ensure_archive
from diarisation.io.rttm import Segment, read_rttm, write_rttm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/ami_fa.yaml")
    cfg = load_yaml(ap.parse_args().config)
    ami, wcfg = load_yaml(cfg["lists_from"]), load_yaml(cfg["words"])
    clone = Path(cfg["local_clone"])
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != cfg["source"]["commit"]:
        print(f"ABORT: {clone} is at {head or 'nothing'}, config pins {cfg['source']['commit']}", file=sys.stderr)
        return 2
    agents = agent_map(ensure_archive(wcfg, load_paths()["cache_root"]))
    summary = {}
    for part in cfg["partitions"]:
        uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()
        n_seg, fa_s, ow_s = 0, 0.0, 0.0
        for uri in uris:
            src = clone / cfg["source"]["path"].format(partition=part, uri=uri)
            if not src.exists():
                print(f"ABORT: {src} missing", file=sys.stderr)
                return 2
            segs = read_rttm(src).get(uri, [])
            out = []
            for s in segs:
                m, letter = s.speaker.rsplit(".", 1)
                if m != uri or letter not in agents[uri]:
                    print(f"ABORT {uri}: speaker {s.speaker} not in meetings.xml", file=sys.stderr)
                    return 2
                out.append(Segment(s.start, s.end, agents[uri][letter]))
            ow = read_rttm(REPO_ROOT / ami["out_dir"] / "only_words" / "rttm" / part / f"{uri}.rttm")[uri]
            if {s.speaker for s in out} != {s.speaker for s in ow}:
                print(f"ABORT {uri}: speakers {sorted({s.speaker for s in out})} != only_words "
                      f"{sorted({s.speaker for s in ow})}", file=sys.stderr)
                return 2
            write_rttm(REPO_ROOT / cfg["out_dir"] / part / f"{uri}.rttm", uri, out)
            n_seg += len(out)
            fa_s += sum(s.duration for s in out)
            ow_s += sum(s.duration for s in ow)
        summary[part] = {"meetings": len(uris), "segments": n_seg, "speaker_s": round(fa_s, 3),
                         "only_words_speaker_s": round(ow_s, 3), "ratio": round(fa_s / ow_s, 4)}
    man = REPO_ROOT / cfg["out_dir"] / "MANIFEST.json"
    man.write_text(json.dumps({"source": cfg["source"], "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
