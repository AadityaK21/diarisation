"""Paired comparison: optimal (md-eval) mapping vs mapping fixed on single-speaker frames.

Reads ONLY the two experiments' per_file.csv, plus writes mappings.csv (one
row per file and system: both mappings and whether they differ). All numbers
are pooled over files with a paired bootstrap (same resampled meetings).
"""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.scoring.der import frame_arrays
from diarisation.stats.bootstrap import of_sums

COMP = ["scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimal", default="configs/experiments/decomp_pyannote31.yaml")
    ap.add_argument("--fixed", default="configs/experiments/decomp_pyannote31_singlemap.yaml")
    ap.add_argument("--out", default="results/mapping_comparison")
    args = ap.parse_args()
    a_cfg, b_cfg = load_yaml(args.optimal), load_yaml(args.fixed)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    bs = a_cfg["bootstrap"]

    def load(cfg):
        rows = list(csv.DictReader(open(REPO_ROOT / cfg["out_dir"] / "per_file.csv")))
        return {(r["system"], r["uri"], float(r["collar"]), int(r["exclude_overlap"]), r["region"]): r for r in rows}
    A, B = load(a_cfg), load(b_cfg)

    # Mappings per file, recorded so the "how often does it change" count is checkable.
    ami = load_yaml("configs/data/ami.yaml")
    mrows = []
    for system, spec in a_cfg["systems"].items():
        for uri in sorted({k[1] for k in A if k[0] == system}):
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / "only_words" / "rttm" / "test" / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / "test" / f"{uri}.uem")[uri]
            hyp = read_rttm(REPO_ROOT / spec["hyp_dir"] / f"{uri}.rttm")[uri]
            m_all = frame_arrays(ref, hyp, uem, 0.0, False, a_cfg["frame_step"], "all").mapping
            m_one = frame_arrays(ref, hyp, uem, 0.0, False, a_cfg["frame_step"], "single").mapping
            changed = sorted(r for r in set(m_all) | set(m_one) if m_all.get(r) != m_one.get(r))
            mrows.append({"system": system, "uri": uri, "mapping_all": ";".join(f"{k}>{v}" for k, v in sorted(m_all.items())),
                          "mapping_single": ";".join(f"{k}>{v}" for k, v in sorted(m_one.items())),
                          "n_ref_speakers_remapped": len(changed), "remapped": ";".join(changed)})
    with open(out / "mappings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrows[0]))
        w.writeheader(); w.writerows(mrows)

    res = []
    for system in a_cfg["systems"]:
        uris = sorted({k[1] for k in A if k[0] == system})
        for conv, spec in a_cfg["conventions"].items():
            c, x = float(spec["collar"]), int(spec["exclude_overlap"])
            for region in ("all", "single", "overlap"):
                cols = {}
                for tag, T in (("opt", A), ("fix", B)):
                    for k in COMP:
                        cols[f"{tag}_{k}"] = [float(T[(system, u, c, x, region)][k]) for u in uris]
                if sum(cols["opt_scored_speaker_s"]) == 0:
                    continue
                err = lambda s, t: s[f"{t}_missed_s"] + s[f"{t}_false_alarm_s"] + s[f"{t}_confusion_s"]  # noqa: E731
                stats = {
                    "der_optimal": lambda s: err(s, "opt") / s["opt_scored_speaker_s"],
                    "der_fixed": lambda s: err(s, "fix") / s["fix_scored_speaker_s"],
                    "der_fixed_minus_optimal": lambda s: err(s, "fix") / s["fix_scored_speaker_s"] - err(s, "opt") / s["opt_scored_speaker_s"],
                    "conf_fixed_minus_optimal": lambda s: (s["fix_confusion_s"] - s["opt_confusion_s"]) / s["opt_scored_speaker_s"],
                }
                for name, f in stats.items():
                    e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
                    res.append({"system": system, "convention": conv, "region": region, "statistic": name,
                                "n_files": len(uris), "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}",
                                "n_resamples": bs["n_resamples"], "seed": bs["seed"]})
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    n_changed = sum(1 for r in mrows if r["n_ref_speakers_remapped"] > 0)
    print(f"{n_changed} of {len(mrows)} file-system pairs have a different mapping -> {out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
