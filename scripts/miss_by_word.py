"""Which single-speaker words does a system miss? (decisions S1)

For every reference word (AMI manual words, true times), the part of its
duration that lies inside the UEM with exactly one reference speaker active is
"missed" where the system outputs no speaker at all (1 ms frames, the scorer's
framing). Words are grouped by the length of the reference turn they sit in
(as in X3: the speaker's merged reference region holding the word's midpoint)
and by their own duration. Outputs per_file.csv (seconds per group) and
aggregate.csv (pooled missed share with bootstrap intervals over meetings).

Also checks the reference convention: the share of single-speaker reference
time covered by the active speaker's own words (1.0 = the reference marks
exactly the word time as speech)."""

import argparse
import csv
import sys

import numpy as np

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.intervals import per_speaker_union
from diarisation.io.rttm import Segment, read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.der import _activity, _Grid, _uem_weight  # the scorer's own framing
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from overlap_words import region_of  # noqa: E402


def dur_bin(d, edges):
    lo = 0.0
    for e in edges:
        if d < e:
            return f"{lo:g}-{e:g} s"
        lo = e
    return f">={edges[-1]:g} s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/miss_by_word.yaml")
    cfg = load_yaml(ap.parse_args().config)
    ami, wcfg, bs = load_yaml(cfg["dataset"]), load_yaml(cfg["words"]), cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"], packages=("numpy",))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part, step, short, edges = cfg["partition"], cfg["frame_step"], cfg["short_turn_max_words"], cfg["duration_edges_s"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()

    rows, coverage = [], {"single_s": 0.0, "covered_s": 0.0}
    for uri in uris:
        ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
        uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
        words = read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv")
        hyps = {s: read_rttm(REPO_ROOT / sc["hyp_dir"] / f"{uri}.rttm").get(uri, []) for s, sc in cfg["systems"].items()}
        end = max([e for _, e in uem] + [s.end for s in ref] + [w.end for w in words]
                  + [s.end for h in hyps.values() for s in h] + [0.0])
        grid = _Grid(step, int(np.ceil(end / step)) + 1)
        sec = _uem_weight(uem, step, grid.n) * step
        r_act = _activity(ref, grid)
        n_ref = sum((a.astype(np.int16) for a in r_act.values()), np.zeros(grid.n, dtype=np.int16))
        single = n_ref == 1

        by_spk = per_speaker_union(ref)
        turn_words = {}
        for w in words:
            reg = region_of(by_spk.get(w.speaker, []), 0.5 * (w.start + w.end))
            turn_words[(w.speaker, reg)] = turn_words.get((w.speaker, reg), 0) + 1
        w_act = _activity([Segment(w.start, w.end, w.speaker) for w in words], grid)
        cov = np.zeros(grid.n, dtype=bool)
        for spk, a in r_act.items():
            if spk in w_act:
                cov |= a & w_act[spk]
        coverage["single_s"] += sec[single].sum()
        coverage["covered_s"] += sec[single & cov].sum()

        none = {s: sum((a.astype(np.int16) for a in _activity(h, grid).values()),
                       np.zeros(grid.n, dtype=np.int16)) == 0 for s, h in hyps.items()}
        acc = {}
        for w in words:
            i0, i1 = grid.idx(w.start), grid.idx(w.end)
            if i1 <= i0:
                continue
            m = single[i0:i1]
            t = sec[i0:i1][m].sum()
            if t <= 0:
                continue
            n_turn = turn_words.get((w.speaker, region_of(by_spk.get(w.speaker, []), 0.5 * (w.start + w.end))), 1)
            groups = [("all", "all"),
                      ("turn_length", f"1-{short} words" if n_turn <= short else f">{short} words"),
                      ("duration", dur_bin(w.end - w.start, edges))]
            for s in hyps:
                missed = sec[i0:i1][m & none[s][i0:i1]].sum()
                for g in groups:
                    a = acc.setdefault((s,) + g, [0.0, 0.0, 0])
                    a[0] += t; a[1] += missed; a[2] += 1
        for (s, gt, g), (t, missed, n) in sorted(acc.items()):
            rows.append({"system": s, "uri": uri, "group_type": gt, "group": g, "n_words": n,
                         "single_s": f"{t:.3f}", "missed_s": f"{missed:.3f}"})
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    back = list(csv.DictReader(open(out / "per_file.csv")))
    agg = []
    for key in sorted({(r["system"], r["group_type"], r["group"]) for r in back}):
        sel = [r for r in back if (r["system"], r["group_type"], r["group"]) == key]
        cols = {k: [float(r[k]) for r in sel] for k in ("single_s", "missed_s")}
        e = of_sums(cols, lambda s: s["missed_s"] / s["single_s"], bs["n_resamples"], bs["ci_level"], bs["seed"])
        agg.append({"system": key[0], "group_type": key[1], "group": key[2], "n_files": len(sel),
                    "n_words": sum(int(r["n_words"]) for r in sel), "single_s": f"{sum(cols['single_s']):.3f}",
                    "missed_share": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    share = coverage["covered_s"] / coverage["single_s"]
    log.finish(out, single_speaker_time_covered_by_own_words=round(share, 6))
    print(f"single-speaker reference time covered by the speaker's own words: {share:.4f}")
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
