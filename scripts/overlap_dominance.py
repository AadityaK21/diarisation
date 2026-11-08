"""Two-speaker overlap shown as one speaker: louder or quieter talker?
(config comments define it). Per meeting counts of samples where the shown
speaker is the louder / the quieter of the two / neither; rates with
bootstrap CIs over meetings, read from per_file.csv."""

import argparse
import bisect
import csv
import math
import sys

import numpy as np

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami_words import ensure_archive, headset_channels
from diarisation.intervals import per_speaker_union
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.der import frame_arrays
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from overlap_words import Energy  # noqa: E402


def active(regions, t):
    i = bisect.bisect_right(regions, (t, math.inf)) - 1
    return i >= 0 and regions[i][0] <= t < regions[i][1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/overlap_dominance_pyannote31.yaml")
    cfg = load_yaml(ap.parse_args().config)
    ami, wcfg, bs = load_yaml(cfg["dataset"]), load_yaml(cfg["words"]), cfg["bootstrap"]
    paths = load_paths()
    chans = headset_channels(ensure_archive(wcfg, paths["cache_root"]))
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"], packages=("numpy", "soundfile"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part, hop, half = cfg["partition"], cfg["hop"], cfg["window"] / 2
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()
    rows = []
    for uri in uris:
        ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
        uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
        rby = per_speaker_union(ref)
        audio = paths["audio_root"] / "amicorpus" / uri / "audio"
        energy = {spk: Energy(audio / f"{uri}.Headset-{ch}.wav") for spk, ch in chans[uri].items()}
        for system, scfg in cfg["systems"].items():
            hyp = read_rttm(REPO_ROOT / scfg["hyp_dir"] / f"{uri}.rttm")[uri]
            hby = per_speaker_union(hyp)
            to_ref = {h: r for r, h in frame_arrays(ref, hyp, uem, 0.0, False, cfg["frame_step"], "all").mapping.items()}
            n = {"louder": 0, "quieter": 0, "neither": 0, "both_shown": 0, "none_shown": 0}
            for a, b in uem:
                for t in np.arange(a + half, b - half, hop):
                    ra = [r for r, reg in rby.items() if active(reg, t)]
                    if len(ra) != 2:
                        continue
                    ha = [h for h, reg in hby.items() if active(reg, t)]
                    if len(ha) == 0:
                        n["none_shown"] += 1
                        continue
                    if len(ha) >= 2:
                        n["both_shown"] += 1
                        continue
                    shown = to_ref.get(ha[0])
                    if shown not in ra:
                        n["neither"] += 1
                        continue
                    other = ra[1] if ra[0] == shown else ra[0]
                    e_s, e_o = energy[shown](t - half, t + half), energy[other](t - half, t + half)
                    n["louder" if e_s >= e_o else "quieter"] += 1
            rows.append({"system": system, "uri": uri, **n})
        print(f"  {uri}: " + "; ".join(f"{r['system']} {r['louder']}/{r['quieter']}" for r in rows if r["uri"] == uri), flush=True)
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    back = list(csv.DictReader(open(out / "per_file.csv")))
    agg = []
    for system in cfg["systems"]:
        sel = [r for r in back if r["system"] == system]
        cols = {k: [float(r[k]) for r in sel] for k in ("louder", "quieter", "neither", "both_shown", "none_shown")}
        tot = lambda s: s["louder"] + s["quieter"] + s["neither"] + s["both_shown"] + s["none_shown"]  # noqa: E731
        stats = {"shown_is_louder_given_one_of_pair": lambda s: s["louder"] / (s["louder"] + s["quieter"]),
                 "both_shown": lambda s: s["both_shown"] / tot(s),
                 "one_shown_of_pair": lambda s: (s["louder"] + s["quieter"]) / tot(s),
                 "other_or_none_shown": lambda s: (s["neither"] + s["none_shown"]) / tot(s)}
        for name, f in stats.items():
            e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
            n_samples = int(tot({k: sum(v) for k, v in cols.items()}))
            agg.append({"system": system, "statistic": name, "n_files": len(sel), "n_samples": n_samples,
                        "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    log.finish(out, n_rows=len(back))
    for a in agg:
        print(f"{a['system']:20s} {a['statistic']:36s} {float(a['point'])*100:6.1f} [{float(a['lo'])*100:.1f}, {float(a['hi'])*100:.1f}]")
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
