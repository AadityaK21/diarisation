"""Reproduction check (decisions P2): our own CPU run of pyannote 3.1 against
the authors' published AMI test outputs, on the meetings our run has finished.

Per meeting and condition: Full-convention DER (no collar, overlap scored,
1 ms frames, optimal mapping) of both outputs against the reference, and the
disagreement between the two outputs, scored as DER of our output with the
published output taken as the reference, inside the reference UEM.

Outputs per_file.csv, aggregate.csv (pooled over the finished meetings; the
paired difference ours - published has a bootstrap interval over meetings,
which means little until most meetings are done) and run.json."""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.conditioned import decompose
from diarisation.scoring.der import frame_arrays
from diarisation.stats.bootstrap import of_sums

COMP = ["scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"]


def score(ref, hyp, uem, step):
    r = decompose(frame_arrays(ref, hyp, uem, 0.0, False, step, "all"))["all"]
    return {k: getattr(r, k) for k in COMP}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/reproduction_pyannote31.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ami, bs, step = load_yaml(cfg["dataset"]), cfg["bootstrap"], cfg["frame_step"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"], packages=("numpy",))
    out = REPO_ROOT / cfg["out_dir"]
    part = cfg["partition"]
    root = REPO_ROOT / ami["out_dir"]
    uris = (root / "lists" / f"{part}.txt").read_text().split()

    rows, done = [], {}
    for cond, spec in cfg["conditions"].items():
        ours_dir, pub_dir = REPO_ROOT / spec["ours_dir"], REPO_ROOT / spec["published_dir"]
        done[cond] = [u for u in uris if (ours_dir / f"{u}.rttm").exists()]
        for uri in done[cond]:
            ref = read_rttm(root / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(root / "uem" / part / f"{uri}.uem")[uri]
            ours = read_rttm(ours_dir / f"{uri}.rttm").get(uri, [])
            pub = read_rttm(pub_dir / f"{uri}.rttm").get(uri, [])
            row = {"condition": cond, "uri": uri,
                   "n_speakers_ours": len({s.speaker for s in ours}),
                   "n_speakers_published": len({s.speaker for s in pub})}
            for name, (a, b) in {"ours": (ref, ours), "published": (ref, pub),
                                 "ours_vs_published": (pub, ours)}.items():
                row.update({f"{name}_{k}": f"{v:.6f}" for k, v in score(a, b, uem, step).items()})
            rows.append(row)
    if not rows:
        print("ABORT: no finished meetings in any ours_dir", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- everything below reads only per_file.csv ----------------
    back = list(csv.DictReader(open(out / "per_file.csv")))
    err = lambda s, p: (s[f"{p}_missed_s"] + s[f"{p}_false_alarm_s"] + s[f"{p}_confusion_s"]) / s[f"{p}_scored_speaker_s"]  # noqa: E731
    stats = {
        "der_ours": lambda s: err(s, "ours"),
        "der_published": lambda s: err(s, "published"),
        "der_ours_minus_published": lambda s: err(s, "ours") - err(s, "published"),
        "disagreement": lambda s: err(s, "ours_vs_published"),
        "disagreement_missed": lambda s: s["ours_vs_published_missed_s"] / s["ours_vs_published_scored_speaker_s"],
        "disagreement_false_alarm": lambda s: s["ours_vs_published_false_alarm_s"] / s["ours_vs_published_scored_speaker_s"],
        "disagreement_confusion": lambda s: s["ours_vs_published_confusion_s"] / s["ours_vs_published_scored_speaker_s"],
    }
    agg = []
    for cond in cfg["conditions"]:
        sel = [r for r in back if r["condition"] == cond]
        if not sel:
            continue
        keys = [k for k in sel[0] if k.endswith("_s")]
        cols = {k: [float(r[k]) for r in sel] for k in keys}
        for name, f in stats.items():
            e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
            agg.append({"condition": cond, "statistic": name, "n_files": len(sel),
                        "n_files_total": len(uris), "point": f"{e.point:.6f}",
                        "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    log.finish(out, finished={c: u for c, u in done.items()})
    for c, u in done.items():
        print(f"{c}: {len(u)}/{len(uris)} meetings finished")
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
