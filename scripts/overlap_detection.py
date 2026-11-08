"""Overlap detection by a diarisation system: speaker-count confusion against
the reference, per meeting, then recall/precision of overlap with bootstrap
CIs over meetings (aggregates read only from per_file.csv)."""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.stats.bootstrap import of_sums
from diarisation.stats.count_confusion import capped, count_confusion


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/overlap_detection_pyannote31.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ami, bs, cap = load_yaml(cfg["dataset"]), cfg["bootstrap"], cfg["count_cap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"], packages=("numpy",))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part = cfg["partition"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()
    keys = [(r, h) for r in range(cap + 1) for h in range(cap + 1)]
    col = lambda r, h: f"t_ref{r}_sys{h}"  # noqa: E731

    rows = []
    for system, scfg in cfg["systems"].items():
        for uri in uris:
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
            hyp = read_rttm(REPO_ROOT / scfg["hyp_dir"] / f"{uri}.rttm")[uri]
            conf = capped(count_confusion(ref, hyp, uem), cap)
            rows.append({"system": system, "uri": uri, **{col(r, h): f"{conf.get((r, h), 0.0):.6f}" for r, h in keys}})
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    back = list(csv.DictReader(open(out / "per_file.csv")))
    agg = []
    for system in cfg["systems"]:
        sel = [r for r in back if r["system"] == system]
        cols = {col(r, h): [float(x[col(r, h)]) for x in sel] for r, h in keys}
        ov_ref = lambda s: sum(s[col(r, h)] for r in range(2, cap + 1) for h in range(cap + 1))  # noqa: E731
        ov_sys = lambda s: sum(s[col(r, h)] for r in range(cap + 1) for h in range(2, cap + 1))  # noqa: E731
        both = lambda s: sum(s[col(r, h)] for r in range(2, cap + 1) for h in range(2, cap + 1))  # noqa: E731
        stats = {
            "overlap_recall": lambda s: both(s) / ov_ref(s),
            "overlap_precision": lambda s: both(s) / ov_sys(s),
            "ref_overlap_shown_as_one_speaker": lambda s: sum(s[col(r, 1)] for r in range(2, cap + 1)) / ov_ref(s),
            "ref_overlap_shown_as_no_speech": lambda s: sum(s[col(r, 0)] for r in range(2, cap + 1)) / ov_ref(s),
            "sys_overlap_share_of_speech": lambda s: ov_sys(s) / sum(s[col(r, h)] for r in range(cap + 1) for h in range(1, cap + 1)),
            "ref_overlap_share_of_speech": lambda s: ov_ref(s) / sum(s[col(r, h)] for r in range(1, cap + 1) for h in range(cap + 1)),
        }
        for name, f in stats.items():
            e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
            agg.append({"system": system, "statistic": name, "n_files": len(sel), "point": f"{e.point:.6f}",
                        "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
        tot = {k: sum(v) for k, v in cols.items()}
        for r, h in keys:
            agg.append({"system": system, "statistic": f"seconds_ref{r}_sys{h}", "n_files": len(sel),
                        "point": f"{tot[col(r, h)]:.3f}", "lo": "", "hi": ""})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    log.finish(out, n_rows=len(back))
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
