"""Write the AMI word-level reference and check it against the diarisation reference."""

import argparse
import json
from bisect import bisect_right

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data import ami_words
from diarisation.intervals import per_speaker_union
from diarisation.io.rttm import read_rttm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/ami_words.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ami = load_yaml(cfg["lists_from"])
    summary = ami_words.prepare(cfg, ami, load_paths()["cache_root"])

    # Consistency: each word's midpoint should lie in its own speaker's RTTM turns.
    for part in cfg["partitions"]:
        inside = total = 0
        unknown = set()
        for uri in (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split():
            ref = per_speaker_union(read_rttm(REPO_ROOT / ami["out_dir"] / "only_words" / "rttm" / part / f"{uri}.rttm")[uri])
            for w in ami_words.read_tsv(REPO_ROOT / cfg["out_dir"] / part / f"{uri}.tsv"):
                total += 1
                regions = ref.get(w.speaker)
                if regions is None:
                    unknown.add((uri, w.speaker)); continue
                mid = 0.5 * (w.start + w.end)
                i = bisect_right(regions, (mid, float("inf"))) - 1
                inside += i >= 0 and regions[i][0] <= mid <= regions[i][1]
        summary[part]["word_midpoint_in_own_rttm_turn"] = round(inside / total, 6)
        summary[part]["speakers_missing_from_rttm"] = sorted(map(list, unknown))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
