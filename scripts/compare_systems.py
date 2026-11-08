"""Paired comparison of two diarisation systems on the same meetings
(decisions S1): DER by region and its components (Full and Fair), overlap
detection, and speaker-attributed WER with the same Whisper words.

For every statistic: the baseline, the contender, their paired difference
(contender - baseline) with a bootstrap interval over meetings, and the
number of meetings where the contender is lower, with an exact sign test
(ties dropped). Reads only per_file.csv files written by other scripts."""

import argparse
import csv
import sys
from collections import defaultdict

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.runlog import RunLog
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from consistency import sign_test  # noqa: E402

COMP = ("scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s")
WER_KEYS = ("n_ref", "sub", "del", "ins")


def load(path):
    return list(csv.DictReader(open(REPO_ROOT / path)))


def compare(cols_b, cols_c, uris, f, bs):
    """f maps a dict of summed columns to a number; columns are paired by meeting."""
    cols = {**{f"b|{k}": v for k, v in cols_b.items()}, **{f"c|{k}": v for k, v in cols_c.items()}}
    pick = lambda s, side: {k[2:]: v for k, v in s.items() if k.startswith(side + "|")}  # noqa: E731
    one = lambda side: lambda s: f(pick(s, side))  # noqa: E731
    diff = lambda s: f(pick(s, "c")) - f(pick(s, "b"))  # noqa: E731
    est = {name: of_sums(cols, g, bs["n_resamples"], bs["ci_level"], bs["seed"])
           for name, g in (("baseline", one("b")), ("contender", one("c")), ("difference", diff))}
    per = []
    for i in range(len(uris)):
        try:
            per.append(diff({k: v[i] for k, v in cols.items()}))
        except ZeroDivisionError:      # e.g. a meeting with no 3-speaker overlap
            per.append(None)
    lower = sum(x is not None and x < 0 for x in per)
    higher = sum(x is not None and x > 0 for x in per)
    row = {}
    for name, e in est.items():
        row[name] = f"{e.point:.6f}"
        if name == "difference":
            row["diff_lo"], row["diff_hi"] = f"{e.lo:.6f}", f"{e.hi:.6f}"
    undefined = sum(x is None for x in per)
    row.update({"n_files": len(uris), "n_contender_lower": lower, "n_ties": len(uris) - lower - higher - undefined,
                "n_undefined": undefined,
                "p_sign_test": f"{sign_test(lower, lower + higher):.3g}"})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/compare_sortformer_pyannote.yaml")
    cfg = load_yaml(ap.parse_args().config)
    bs = cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"], packages=("numpy",))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    # meetings in the corpus list order, as in every other script, so that the
    # seeded bootstrap draws the same meetings as theirs
    ami = load_yaml(cfg["dataset"])
    order = {u: i for i, u in enumerate((REPO_ROOT / ami["out_dir"] / "lists" / f"{cfg['partition']}.txt").read_text().split())}
    ordered = lambda us: sorted(us, key=lambda u: order[u])  # noqa: E731

    der = defaultdict(dict)   # (system, collar, excl, region) -> {uri: row}
    for r in load(cfg["der"]):
        der[(r["system"], float(r["collar"]), int(r["exclude_overlap"]), r["region"])][r["uri"]] = r
    od = defaultdict(dict)    # system -> {uri: row}
    for path in cfg["overlap_detection"]:
        for r in load(path):
            od[r["system"]][r["uri"]] = r
    wer = defaultdict(dict)   # (condition, measure, region) -> {uri: row}
    for r in load(cfg["wer"]):
        wer[(r["condition"], r["measure"], r["region"])][r["uri"]] = r

    rows = []
    for p in cfg["pairs"]:
        b, c = p["baseline"], p["contender"]
        for conv, spec in cfg["conventions"].items():
            for region in ("all", "single", "overlap"):
                kb = (b, float(spec["collar"]), int(spec["exclude_overlap"]), region)
                kc = (c, float(spec["collar"]), int(spec["exclude_overlap"]), region)
                uris = ordered(set(der[kb]) & set(der[kc]))
                if len(uris) != len(der[kb]) or len(uris) != len(der[kc]):
                    print(f"ABORT {p['name']}: the two systems cover different meetings", file=sys.stderr)
                    return 2
                cb = {k: [float(der[kb][u][k]) for u in uris] for k in COMP}
                cc = {k: [float(der[kc][u][k]) for u in uris] for k in COMP}
                for name, f in {
                    "der": lambda s: (s["missed_s"] + s["false_alarm_s"] + s["confusion_s"]) / s["scored_speaker_s"],
                    "missed": lambda s: s["missed_s"] / s["scored_speaker_s"],
                    "false_alarm": lambda s: s["false_alarm_s"] / s["scored_speaker_s"],
                    "confusion": lambda s: s["confusion_s"] / s["scored_speaker_s"],
                }.items():
                    rows.append({"pair": p["name"], "family": f"der_{conv}", "region": region, "statistic": name,
                                 **compare(cb, cc, uris, f, bs)})

        uris = ordered(set(od[b]) & set(od[c]))
        keys = [k for k in od[b][uris[0]] if k.startswith("t_ref")]
        cap = max(int(k.split("_sys")[1]) for k in keys)
        col = lambda r, h: f"t_ref{r}_sys{h}"  # noqa: E731
        ov_ref = lambda s: sum(s[col(r, h)] for r in range(2, cap + 1) for h in range(cap + 1))  # noqa: E731
        ov_sys = lambda s: sum(s[col(r, h)] for r in range(cap + 1) for h in range(2, cap + 1))  # noqa: E731
        both = lambda s: sum(s[col(r, h)] for r in range(2, cap + 1) for h in range(2, cap + 1))  # noqa: E731
        cb = {k: [float(od[b][u][k]) for u in uris] for k in keys}
        cc = {k: [float(od[c][u][k]) for u in uris] for k in keys}
        for name, f in {
            "overlap_recall": lambda s: both(s) / ov_ref(s),
            "overlap_precision": lambda s: both(s) / ov_sys(s),
            "ref_overlap_shown_as_one_speaker": lambda s: sum(s[col(r, 1)] for r in range(2, cap + 1)) / ov_ref(s),
            "ref3plus_shown_as_3plus": lambda s: s[col(cap, cap)] / sum(s[col(cap, h)] for h in range(cap + 1)),
        }.items():
            rows.append({"pair": p["name"], "family": "overlap_detection", "region": "overlap", "statistic": name,
                         **compare(cb, cc, uris, f, bs)})

        cond, wb, wc = p["wer_condition"], p["wer_baseline"], p["wer_contender"]
        for region in ("all", "single", "overlap"):
            uris = ordered(wer[(cond, f"cpwer_{wb}", region)])
            pick = lambda m: {f"{m}|{k}": [float(wer[(cond, m, region)][u][k]) for u in uris] for k in WER_KEYS}  # noqa: E731
            w = lambda s, m: (s[f"{m}|sub"] + s[f"{m}|del"] + s[f"{m}|ins"]) / s[f"{m}|n_ref"]  # noqa: E731
            base = {**pick(f"cpwer_{wb}"), **pick("cpwer_oracle")}
            cont = {**{k.replace(f"cpwer_{wc}|", f"cpwer_{wb}|"): v for k, v in pick(f"cpwer_{wc}").items()},
                    **pick("cpwer_oracle")}
            for name, f in {
                "cpwer": lambda s: w(s, f"cpwer_{wb}"),
                "part_diarisation": lambda s: w(s, f"cpwer_{wb}") - w(s, "cpwer_oracle"),
            }.items():
                rows.append({"pair": p["name"], "family": "wer", "region": region, "statistic": name,
                             **compare(base, cont, uris, f, bs)})

    with open(out / "contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    log.finish(out, n_rows=len(rows))
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
