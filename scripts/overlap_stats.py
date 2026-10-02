"""Step 2: per-file reference overlap table and bootstrap aggregates."""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.stats import overlap


def dataset_files(ds_cfg, annotation):
    """Yield (partition, uri, rttm_path, uem_path) for one dataset/annotation."""
    root = REPO_ROOT / ds_cfg["out_dir"]
    if ds_cfg["name"] == "ami":
        for part in ds_cfg["partitions"]:
            for uri in (root / "lists" / f"{part}.txt").read_text().split():
                yield part, uri, root / annotation / "rttm" / part / f"{uri}.rttm", \
                    root / "uem" / part / f"{uri}.uem"
    else:
        for uri in (root / "lists" / "all.txt").read_text().split():
            yield "all", uri, root / "rttm" / f"{uri}.rttm", root / "uem" / f"{uri}.uem"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stats/overlap.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    # Hash the dataset configs too: a changed partition must change the run id.
    full = {"stats": cfg, "datasets": {d["config"]: load_yaml(d["config"]) for d in cfg["datasets"]}}
    bs = cfg["bootstrap"]
    log = RunLog(cfg["experiment"], full, seed=bs["seed"])
    out_dir = REPO_ROOT / cfg["out_dir"]

    rows = []
    for d in cfg["datasets"]:
        ds_cfg = full["datasets"][d["config"]]
        for part, uri, rttm_p, uem_p in dataset_files(ds_cfg, d["annotation"]):
            if not rttm_p.exists():
                print(f"missing {rttm_p}; run the prepare step first", file=sys.stderr)
                return 1
            segs = read_rttm(rttm_p).get(uri, [])
            uem = read_uem(uem_p)[uri]
            rows.append({"dataset": ds_cfg["name"], "annotation": d["annotation"],
                         "partition": part, "uri": uri, **overlap.file_row(segs, uem)})

    per_file = out_dir / "per_file.csv"
    overlap.write_per_file(per_file, rows)
    agg = overlap.aggregate_from_csv(per_file, bs["n_resamples"], bs["ci_level"], bs["seed"])
    overlap.write_aggregate(out_dir / "aggregate.csv", agg)
    log.finish(out_dir, n_files=len(rows))

    print(f"run {log.short_hash}  seed {bs['seed']}  -> {out_dir.relative_to(REPO_ROOT)}/")
    with open(out_dir / "aggregate.csv") as fh:
        for r in csv.DictReader(fh):
            print(f"{r['dataset']:12} {r['annotation']:21} {r['partition']:5} n={r['n_files']:>3} "
                  f"{float(r['hours_uem']):6.1f} h  overlap/speech {100*float(r['overlap_time_frac']):5.1f}% "
                  f"[{100*float(r['overlap_time_frac_lo'] or 'nan'):.1f}, {100*float(r['overlap_time_frac_hi'] or 'nan'):.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
