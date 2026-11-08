"""Step 6b diagnostics: how much of the speech does single-channel Whisper output?

For each run of the attribution-ASR config: output words per reference word
in each region (a word's region = reference speaker count at its midpoint),
and which reference words are deleted under the cpWER pairing. Reads the
same config as attribution_asr.py; writes <out_dir>/diagnostics/.
"""

import argparse
import csv
import sys
from collections import Counter

from rapidfuzz.distance import Levenshtein

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import assign
from diarisation.scoring.wer import RegionLookup, cp_assignment, tokens
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from attribution_asr import asr_words  # noqa: E402

REGIONS = ("single", "overlap")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/attr_asr_whisper_turbo.yaml")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--top", type=int, default=15, help="how many most-deleted words to list")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.runs:
        cfg["runs"] = [r for r in cfg["runs"] if r["name"] in args.runs]
    ami, wcfg, bs = load_yaml(cfg["dataset"]), load_yaml(cfg["words"]), cfg["bootstrap"]
    log = RunLog(cfg["experiment"] + "_diagnostics", {"experiment": cfg, "top": args.top}, seed=bs["seed"],
                 packages=("numpy", "scipy", "rapidfuzz", "whisper-normalizer"))
    out = REPO_ROOT / cfg["out_dir"] / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    part = cfg["partition"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()

    rows, deleted = [], {}
    for run in cfg["runs"]:
        deleted[run["name"]] = {r: Counter() for r in REGIONS}
        for uri in uris:
            asr_p = REPO_ROOT / run["asr_dir"] / f"{uri}.words.tsv"
            if not asr_p.exists():
                continue
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
            hyp = read_rttm(REPO_ROOT / run["hyp_dir"] / f"{uri}.rttm")[uri]
            region = RegionLookup(ref, uem)
            ref_streams = {}
            for w in read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv"):
                ref_streams.setdefault(w.speaker, []).append((w.text, w.start, w.end))
            ref_streams = {k: tokens(v) for k, v in ref_streams.items()}
            words = asr_words(asr_p)
            hyp_streams = {}
            for w, lab in zip(words, assign(words, hyp)):
                hyp_streams.setdefault(lab if lab is not None else "__unassigned__", []).append((w.text, w.start, w.end))
            hyp_streams = {k: tokens(v) for k, v in hyp_streams.items()}

            n_ref, n_hyp = Counter(), Counter()
            for toks in ref_streams.values():
                n_ref.update(region(t) for t in toks)
            for toks in hyp_streams.values():
                n_hyp.update(region(t) for t in toks)
            for r, h in cp_assignment(ref_streams, hyp_streams).items():
                rt = ref_streams[r]
                ht = hyp_streams.get(h, []) if h is not None else []
                for op in Levenshtein.editops([t.text for t in rt], [t.text for t in ht]):
                    if op.tag == "delete" and region(rt[op.src_pos]) in REGIONS:
                        deleted[run["name"]][region(rt[op.src_pos])][rt[op.src_pos].text] += 1
            rows.append({"run": run["name"], "uri": uri,
                         **{f"{r}_n_ref": n_ref[r] for r in REGIONS}, **{f"{r}_n_hyp": n_hyp[r] for r in REGIONS}})
        print(f"  {run['name']}: {sum(1 for x in rows if x['run'] == run['name'])} meetings", flush=True)

    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # aggregates read only per_file.csv
    back = list(csv.DictReader(open(out / "per_file.csv")))
    agg = []
    for run in cfg["runs"]:
        sel = [r for r in back if r["run"] == run["name"]]
        if not sel:
            continue
        for reg in REGIONS:
            cols = {k: [float(r[f"{reg}_{k}"]) for r in sel] for k in ("n_ref", "n_hyp")}
            e = of_sums(cols, lambda s: s["n_hyp"] / s["n_ref"], bs["n_resamples"], bs["ci_level"], bs["seed"])
            agg.append({"run": run["name"], "region": reg, "n_files": len(sel), "n_ref": int(sum(cols["n_ref"])),
                        "n_hyp": int(sum(cols["n_hyp"])), "hyp_per_ref": f"{e.point:.6f}",
                        "hyp_per_ref_lo": f"{e.lo:.6f}", "hyp_per_ref_hi": f"{e.hi:.6f}"})
    with open(out / "output_rate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    with open(out / "top_deleted.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "region", "rank", "word", "n_deleted", "share_of_deletions"])
        for name, by_reg in deleted.items():
            for reg, c in by_reg.items():
                tot = sum(c.values())
                for i, (word, n) in enumerate(c.most_common(args.top), 1):
                    w.writerow([name, reg, i, word, n, f"{n / tot:.6f}" if tot else ""])
    log.finish(out, meetings=len(back))
    print(f"run {log.short_hash} -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
