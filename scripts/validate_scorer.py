"""Step 3: compare scoring/der.py with md-eval-22.pl file by file.

Writes per_file.csv (one row per file x system x collar x overlap mode, both
implementations side by side), aggregate.csv computed from it, and run.json.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm, write_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring import der, mdeval
from diarisation.validation import perturb

COMP = ["scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"]


def files(ds):
    cfg = load_yaml(ds["config"])
    root = REPO_ROOT / cfg["out_dir"]
    for part in ds["partitions"]:
        if cfg["name"] == "ami":
            for uri in (root / "lists" / f"{part}.txt").read_text().split():
                yield cfg["name"], part, uri, root / ds["annotation"] / "rttm" / part / f"{uri}.rttm", \
                    root / "uem" / part / f"{uri}.uem"
        else:
            for uri in (root / "lists" / "all.txt").read_text().split():
                yield cfg["name"], part, uri, root / "rttm" / f"{uri}.rttm", root / "uem" / f"{uri}.uem"


def job(args):
    ds, part, uri, ref_p, uem_p, system, hyp_p, collars, modes, step = args
    ref = read_rttm(ref_p)[uri]
    hyp = read_rttm(hyp_p).get(uri, [])
    uem = read_uem(uem_p)[uri]
    rows = []
    for c in collars:
        for x in modes:
            ours = der.score(ref, hyp, uem, c, x, step)
            ref_impl = mdeval.run(ref_p, hyp_p, uem_p, c, x)
            row = {"dataset": ds, "partition": part, "uri": uri, "system": system,
                   "collar": c, "exclude_overlap": int(x), "off_grid": ours.off_grid}
            for k in COMP:
                row[f"ours_{k}"] = getattr(ours, k)
                row[f"mdeval_{k}"] = getattr(ref_impl, k)
                row[f"absdiff_{k}"] = abs(getattr(ours, k) - getattr(ref_impl, k))
            row["ours_der"], row["mdeval_der"] = ours.der, ref_impl.der
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/validation/scorer.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    log = RunLog(cfg["experiment"], cfg, seed=cfg["seed"], packages=("numpy", "scipy", "PyYAML"))
    work = REPO_ROOT / cfg["work_dir"]
    out = REPO_ROOT / cfg["out_dir"]

    jobs = []
    for ds in cfg["datasets"]:
        for name, part, uri, ref_p, uem_p in files(ds):
            ref = read_rttm(ref_p)[uri]
            uem = read_uem(uem_p)[uri]
            for system, params in cfg["systems"].items():
                hyp = perturb.make(system, params or {}, ref, uem, cfg["seed"], uri)
                hyp_p = work / name / system / f"{uri}.rttm"
                write_rttm(hyp_p, uri, hyp)
                jobs.append((name, part, uri, ref_p, uem_p, system, hyp_p,
                             cfg["collars"], cfg["exclude_overlap"], cfg["frame_step"]))
    print(f"{len(jobs)} file x system jobs, {len(jobs) * len(cfg['collars']) * len(cfg['exclude_overlap'])} comparisons")

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(job, jobs), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["dataset"], r["uri"], r["system"], r["collar"], r["exclude_overlap"])):
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Aggregate strictly from the CSV.
    with open(out / "per_file.csv") as fh:
        back = list(csv.DictReader(fh))
    tol = cfg["tol_seconds"]
    groups = {}
    for r in back:
        groups.setdefault((r["dataset"], r["system"], r["collar"], r["exclude_overlap"]), []).append(r)
    agg = []
    for (ds, sysname, c, x), rs in sorted(groups.items()):
        s = lambda p, k: sum(float(r[f"{p}_{k}"]) for r in rs)  # noqa: E731
        der_o = (s("ours", "missed_s") + s("ours", "false_alarm_s") + s("ours", "confusion_s")) / s("ours", "scored_speaker_s")
        der_m = (s("mdeval", "missed_s") + s("mdeval", "false_alarm_s") + s("mdeval", "confusion_s")) / s("mdeval", "scored_speaker_s")
        maxdiff = max(float(r[f"absdiff_{k}"]) for r in rs for k in COMP)
        offgrid = sum(int(r["off_grid"]) for r in rs)
        agg.append({"dataset": ds, "system": sysname, "collar": c, "exclude_overlap": x, "n_files": len(rs),
                    "pooled_der_ours": f"{der_o:.6f}", "pooled_der_mdeval": f"{der_m:.6f}",
                    "max_abs_component_diff_s": f"{maxdiff:.6f}", "off_grid": offgrid,
                    "pass": int(maxdiff <= tol and offgrid == 0)})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    n_fail = sum(1 for a in agg if not a["pass"])
    worst = max(float(r[f"absdiff_{k}"]) for r in back for k in COMP)
    log.finish(out, n_comparisons=len(back), max_abs_component_diff_s=worst,
               failing_groups=n_fail, tol_seconds=tol)
    print(f"run {log.short_hash}: {len(back)} comparisons, max |ours - md-eval| = {worst:.6f} s, "
          f"{n_fail} of {len(agg)} groups fail (tol {tol} s)")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
