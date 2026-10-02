"""Step 4: overlap-conditioned DER decomposition.

Outputs (all aggregates computed from per_file.csv):
  per_file.csv   file x system x collar x overlap-mode x region, seconds
  aggregate.csv  pooled seconds and rates per group, bootstrap CIs over files
  headline.csv   per system: DER under each convention, error concealed by
                 the forgiving convention, overlap share of error (paired CIs)
  run.json
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.conditioned import REGIONS, decompose, partition_residual
from diarisation.scoring.der import frame_arrays
from diarisation.stats.bootstrap import of_sums
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


def job(a):
    ds, part, uri, ref_p, uem_p, system, spec, synth, cfg = a
    ref = read_rttm(ref_p)[uri]
    uem = read_uem(uem_p)[uri]
    if "synthetic" in spec:
        name = spec["synthetic"]
        hyp = perturb.make(name, synth["systems"][name] or {}, ref, uem, synth["seed"], uri)
    else:
        hyp = read_rttm(REPO_ROOT / spec["hyp_dir"] / f"{uri}.rttm").get(uri, [])
    rows, worst = [], 0.0
    for c in cfg["collars"]:
        for x in cfg["exclude_overlap"]:
            fa = frame_arrays(ref, hyp, uem, c, x, cfg["frame_step"])
            parts = decompose(fa)
            worst = max(worst, partition_residual(parts))
            for region in ("all",) + REGIONS:
                r = parts[region]
                rows.append({"dataset": ds, "partition": part, "uri": uri, "system": system,
                             "collar": c, "exclude_overlap": int(x), "region": region,
                             **{k: getattr(r, k) for k in COMP}, "off_grid": r.off_grid})
    return rows, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/decomp_synthetic.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    synth = load_yaml(cfg["synthetic_source"]) if "synthetic_source" in cfg else None
    bs = cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg, "synthetic": synth}, seed=bs["seed"],
                 packages=("numpy", "scipy", "PyYAML"))
    out = REPO_ROOT / cfg["out_dir"]

    jobs = [(ds_name, part, uri, ref_p, uem_p, system, spec, synth, cfg)
            for ds in cfg["datasets"]
            for ds_name, part, uri, ref_p, uem_p in files(ds)
            for system, spec in cfg["systems"].items()]
    rows, worst = [], 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        for r, w in ex.map(job, jobs):
            rows.extend(r)
            worst = max(worst, w)
    if worst > cfg["partition_tol_seconds"]:
        print(f"ABORT: regions do not sum to total (max residual {worst:.3g} s)", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------------- everything below reads only per_file.csv ----------------
    with open(out / "per_file.csv") as fh:
        back = list(csv.DictReader(fh))
    nb, lvl, seed = bs["n_resamples"], bs["ci_level"], bs["seed"]

    def cols(rs):
        return {k: [float(r[k]) for r in rs] for k in COMP}

    err = lambda s: s["missed_s"] + s["false_alarm_s"] + s["confusion_s"]  # noqa: E731

    groups = {}
    for r in back:
        groups.setdefault((r["dataset"], r["partition"], r["system"], r["collar"],
                           r["exclude_overlap"], r["region"]), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        c = cols(rs)
        sums = {k: sum(v) for k, v in c.items()}
        row = dict(zip(["dataset", "partition", "system", "collar", "exclude_overlap", "region"], key))
        row["n_files"] = len(rs)
        row.update({k: f"{v:.3f}" for k, v in sums.items()})
        # Rates within the region (undefined when the region scores no speaker time).
        for name, f in [("der", lambda s: err(s) / s["scored_speaker_s"]),
                        ("miss_rate", lambda s: s["missed_s"] / s["scored_speaker_s"]),
                        ("fa_rate", lambda s: s["false_alarm_s"] / s["scored_speaker_s"]),
                        ("conf_rate", lambda s: s["confusion_s"] / s["scored_speaker_s"])]:
            if sums["scored_speaker_s"] > 0:
                e = of_sums(c, f, nb, lvl, seed)
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = (f"{x:.6f}" for x in (e.point, e.lo, e.hi))
            else:
                row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = "NA"
        agg.append(row)
    fields = list(agg[0])
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(agg)

    # Headline: paired statistics across conventions, per dataset/partition/system.
    by_file = {}
    for r in back:
        k = (r["dataset"], r["partition"], r["system"])
        by_file.setdefault(k, {}).setdefault(r["uri"], {})[(float(r["collar"]), int(r["exclude_overlap"]), r["region"])] = r
    head = []
    for (ds, part, system), per_uri in sorted(by_file.items()):
        uris = sorted(per_uri)
        col = {}
        for conv, spec in cfg["conventions"].items():
            for region in ("all",) + REGIONS:
                for k in COMP:
                    col[f"{conv}|{region}|{k}"] = [
                        float(per_uri[u][(float(spec["collar"]), int(spec["exclude_overlap"]), region)][k]) for u in uris]
        E = lambda s, conv, reg="all": (s[f"{conv}|{reg}|missed_s"] + s[f"{conv}|{reg}|false_alarm_s"]  # noqa: E731
                                         + s[f"{conv}|{reg}|confusion_s"])
        T = lambda s, conv, reg="all": s[f"{conv}|{reg}|scored_speaker_s"]  # noqa: E731
        stats = {f"der_{conv}": (lambda conv: lambda s: E(s, conv) / T(s, conv))(conv) for conv in cfg["conventions"]}
        stats.update({
            "concealed_full_minus_forgiving": lambda s: E(s, "full") / T(s, "full") - E(s, "forgiving") / T(s, "forgiving"),
            "overlap_share_of_error_full": lambda s: E(s, "full", "overlap") / E(s, "full"),
            "overlap_contribution_to_der_full": lambda s: E(s, "full", "overlap") / T(s, "full"),
            "der_within_overlap_full": lambda s: E(s, "full", "overlap") / T(s, "full", "overlap"),
            "der_within_single_full": lambda s: E(s, "full", "single") / T(s, "full", "single"),
            "overlap_time_kept_by_collar": lambda s: T(s, "fair", "overlap") / T(s, "full", "overlap"),
        })
        for name, f in stats.items():
            e = of_sums(col, f, nb, lvl, seed)
            head.append({"dataset": ds, "partition": part, "system": system, "statistic": name,
                         "n_files": len(uris), "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}",
                         "hi": f"{e.hi:.6f}", "n_resamples": nb, "ci_level": lvl, "seed": seed})
    with open(out / "headline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(head[0]))
        w.writeheader()
        w.writerows(head)

    log.finish(out, n_rows=len(back), max_partition_residual_s=worst,
               off_grid_total=sum(int(r["off_grid"]) for r in back if r["region"] == "all"))
    print(f"run {log.short_hash}: {len(back)} rows, max partition residual {worst:.2e} s -> {out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
