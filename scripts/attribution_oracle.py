"""Step 6a: attribute reference words to speakers via a diarisation output.

Outputs: per_file.csv (counts per file x system x mapping x region),
mappings.csv (both mappings per file), aggregate.csv and headline.csv
(computed only from per_file.csv, bootstrap CIs over meetings), run.json.
"""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import OUTCOMES, mappings, score_words
from diarisation.stats.bootstrap import of_sums

REGIONS = ("all", "single", "overlap")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/attr_oracle_pyannote31.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ami, wcfg = load_yaml(cfg["dataset"]), load_yaml(cfg["words"])
    bs = cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg, "dataset": ami, "words": wcfg}, seed=bs["seed"],
                 packages=("numpy", "scipy", "PyYAML"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part = cfg["partition"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()

    rows, mrows = [], []
    for system, spec in cfg["systems"].items():
        for uri in uris:
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
            hyp = read_rttm(REPO_ROOT / spec["hyp_dir"] / f"{uri}.rttm")[uri]
            words = read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv")
            maps = mappings(ref, hyp, uem, cfg["frame_step"])
            mrows.append({"system": system, "uri": uri,
                          **{m: ";".join(f"{h}>{r}" for h, r in sorted(maps[m].items())) for m in cfg["mappings"]}})
            for m in cfg["mappings"]:
                counts = score_words(words, ref, hyp, uem, maps[m])
                for region in REGIONS:
                    c = counts[region]
                    rows.append({"system": system, "uri": uri, "mapping": m, "region": region,
                                 "n_words": sum(c.values()), **{o: c.get(o, 0) for o in OUTCOMES}})
        print(f"  {system}: {len(uris)} meetings", flush=True)

    for name, data in (("per_file.csv", rows), ("mappings.csv", mrows)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)

    # ---------------- everything below reads only per_file.csv ----------------
    back = list(csv.DictReader(open(out / "per_file.csv")))
    nb, lvl, seed = bs["n_resamples"], bs["ci_level"], bs["seed"]
    idx = {(r["system"], r["mapping"], r["region"], r["uri"]): r for r in back}

    def col(system, m, region, key):
        return [float(idx[(system, m, region, u)][key]) for u in uris]

    agg = []
    for system in cfg["systems"]:
        for m in cfg["mappings"]:
            for region in REGIONS:
                cols = {k: col(system, m, region, k) for k in ("n_words",) + OUTCOMES}
                row = {"system": system, "mapping": m, "region": region, "n_files": len(uris),
                       "n_words": int(sum(cols["n_words"]))}
                stats = {"attribution_error": lambda s: 1 - s["correct"] / s["n_words"]}
                for o in OUTCOMES[1:]:
                    stats[o] = (lambda o: lambda s: s[o] / s["n_words"])(o)
                for name, f in stats.items():
                    e = of_sums(cols, f, nb, lvl, seed)
                    row[name], row[f"{name}_lo"], row[f"{name}_hi"] = f"{e.point:.6f}", f"{e.lo:.6f}", f"{e.hi:.6f}"
                agg.append(row)
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)

    head = []
    for system in cfg["systems"]:
        cols = {}
        for m in cfg["mappings"]:
            for region in REGIONS:
                for k in ("n_words", "correct"):
                    cols[f"{m}|{region}|{k}"] = col(system, m, region, k)
        err = lambda s, m, reg: 1 - s[f"{m}|{reg}|correct"] / s[f"{m}|{reg}|n_words"]  # noqa: E731
        stats = {
            "share_of_words_in_overlap": lambda s: s["optimal|overlap|n_words"] / s["optimal|all|n_words"],
            "share_of_errors_in_overlap_optimal": lambda s: (s["optimal|overlap|n_words"] - s["optimal|overlap|correct"])
                                                             / (s["optimal|all|n_words"] - s["optimal|all|correct"]),
            "gap_majority_minus_optimal_all": lambda s: err(s, "majority_single", "all") - err(s, "optimal", "all"),
            "gap_majority_minus_optimal_overlap": lambda s: err(s, "majority_single", "overlap") - err(s, "optimal", "overlap"),
            "gap_majority_minus_optimal_single": lambda s: err(s, "majority_single", "single") - err(s, "optimal", "single"),
        }
        for name, f in stats.items():
            e = of_sums(cols, f, nb, lvl, seed)
            head.append({"system": system, "statistic": name, "n_files": len(uris), "point": f"{e.point:.6f}",
                         "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}", "n_resamples": nb, "seed": seed})
    with open(out / "headline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(head[0]))
        w.writeheader(); w.writerows(head)
    log.finish(out, n_rows=len(back))
    print(f"run {log.short_hash} -> {out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
