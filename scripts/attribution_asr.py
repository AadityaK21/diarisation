"""Step 6b: cpWER vs fixed-mapping WER on Whisper output, split by region.

Scores every meeting whose ASR output exists (the CPU transcription runs in
the background); run.json records which meetings were included.
"""

import argparse
import csv
import sys

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import assign, mappings
from diarisation.data.ami_words import Word
from diarisation.scoring.wer import (RegionLookup, cp_assignment, merge_by_mapping, score_assignment, tokens)
from diarisation.stats.bootstrap import of_sums


def asr_words(path):
    with open(path, encoding="utf-8") as fh:
        return [Word("", float(r["start"]), float(r["end"]), r["word"]) for r in csv.DictReader(fh, delimiter="\t")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/attr_asr_whisper_turbo.yaml")
    ap.add_argument("--runs", nargs="+", help="score only these runs (names from the config)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.runs:
        unknown = set(args.runs) - {r["name"] for r in cfg["runs"]}
        if unknown:
            ap.error(f"unknown runs: {sorted(unknown)}")
        cfg["runs"] = [r for r in cfg["runs"] if r["name"] in args.runs]
    ami, wcfg = load_yaml(cfg["dataset"]), load_yaml(cfg["words"])
    bs = cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"],
                 packages=("numpy", "scipy", "rapidfuzz", "meeteval", "whisper-normalizer"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part = cfg["partition"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()

    rows, included, worst_meeteval = [], {}, 0
    for run in cfg["runs"]:
        included[run["name"]] = []
        for uri in uris:
            asr_p = REPO_ROOT / run["asr_dir"] / f"{uri}.words.tsv"
            if not asr_p.exists():
                continue
            included[run["name"]].append(uri)
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
            hyp = read_rttm(REPO_ROOT / run["hyp_dir"] / f"{uri}.rttm")[uri]
            region = RegionLookup(ref, uem)

            ref_words = read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv")
            ref_streams = {}
            for w in ref_words:
                ref_streams.setdefault(w.speaker, []).append((w.text, w.start, w.end))
            ref_streams = {k: tokens(v) for k, v in ref_streams.items()}

            words = asr_words(asr_p)
            labels = assign(words, hyp)
            hyp_streams = {}
            for w, lab in zip(words, labels):
                hyp_streams.setdefault(lab if lab is not None else "__unassigned__", []).append((w.text, w.start, w.end))
            hyp_streams = {k: tokens(v) for k, v in hyp_streams.items()}

            cp = cp_assignment(ref_streams, hyp_streams)
            acc_cp = score_assignment(ref_streams, hyp_streams, cp, region)
            maps = mappings(ref, hyp, uem, cfg["frame_step"])
            merged = merge_by_mapping(hyp_streams, maps["majority_single"])
            acc_fx = score_assignment(ref_streams, merged, {r: f"ref:{r}" for r in ref_streams}, region)
            # one-to-one fixed mapping (extra system speakers stay unmatched -> insertions)
            one = {r: h for h, r in maps["onetoone_single"].items()}
            acc_f1 = score_assignment(ref_streams, hyp_streams, {r: one.get(r) for r in ref_streams}, region)

            if cfg.get("validate_against_meeteval"):
                from meeteval.wer.wer.cp import cp_word_error_rate
                me = cp_word_error_rate([" ".join(t.text for t in ref_streams[k]) for k in sorted(ref_streams)],
                                        [" ".join(t.text for t in hyp_streams[k]) for k in sorted(hyp_streams)])
                ours = sum(a["sub"] + a["del"] + a["ins"] for a in acc_cp.values())
                n = sum(a["n_ref"] for a in acc_cp.values())
                worst_meeteval = max(worst_meeteval, abs(ours - me.errors), abs(n - me.length))

            errs = lambda acc: sum(a["sub"] + a["del"] + a["ins"] for a in acc.values())  # noqa: E731
            if errs(acc_f1) < errs(acc_cp):  # cpWER is the minimum over one-to-one pairings
                print(f"ABORT: {uri} one-to-one fixed mapping beats cpWER, impossible", file=sys.stderr)
                return 3
            for name, acc in (("cp", acc_cp), ("fixed", acc_fx), ("fixed1to1", acc_f1)):
                tot = {k: sum(acc[r][k] for r in acc) for k in ("n_ref", "sub", "del", "ins")}
                for reg, a in (("all", tot), ("single", acc["single"]), ("overlap", acc["overlap"])):
                    rows.append({"run": run["name"], "uri": uri, "assignment": name, "region": reg, **a})
        print(f"  {run['name']}: {len(included[run['name']])} meetings", flush=True)

    if worst_meeteval:
        print(f"ABORT: cpWER differs from MeetEval by {worst_meeteval} (errors or length)", file=sys.stderr)
        return 2
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---------------- everything below reads only per_file.csv ----------------
    back = list(csv.DictReader(open(out / "per_file.csv")))
    idx = {(r["run"], r["assignment"], r["region"], r["uri"]): r for r in back}
    agg = []
    for run in cfg["runs"]:
        us = included[run["name"]]
        if not us:
            continue
        for region in ("all", "single", "overlap"):
            cols = {f"{a}_{k}": [float(idx[(run["name"], a, region, u)][k]) for u in us]
                    for a in ("cp", "fixed", "fixed1to1") for k in ("n_ref", "sub", "del", "ins")}
            wer = lambda s, a: (s[f"{a}_sub"] + s[f"{a}_del"] + s[f"{a}_ins"]) / s[f"{a}_n_ref"]  # noqa: E731
            row = {"run": run["name"], "region": region, "n_files": len(us), "n_ref_tokens": int(sum(cols["cp_n_ref"]))}
            for name, f in (("cpwer", lambda s: wer(s, "cp")), ("fixed_wer", lambda s: wer(s, "fixed")),
                            ("fixed_minus_cp", lambda s: wer(s, "fixed") - wer(s, "cp")),
                            ("fixed1to1_wer", lambda s: wer(s, "fixed1to1")),
                            ("fixed1to1_minus_cp", lambda s: wer(s, "fixed1to1") - wer(s, "cp")),
                            ("cp_sub", lambda s: s["cp_sub"] / s["cp_n_ref"]), ("cp_del", lambda s: s["cp_del"] / s["cp_n_ref"]),
                            ("cp_ins", lambda s: s["cp_ins"] / s["cp_n_ref"])):
                e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = f"{e.point:.6f}", f"{e.lo:.6f}", f"{e.hi:.6f}"
            agg.append(row)
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)

    # paired contrasts between runs (same meetings, same ASR words), cpWER
    con = []
    names = {r["name"] for r in cfg["runs"]}
    for a, b in cfg.get("contrasts", []):
        if a not in names or b not in names:
            continue
        us = sorted(set(included[a]) & set(included[b]))
        for region in ("all", "single", "overlap"):
            cols = {f"{tag}_{k}": [float(idx[(x, "cp", region, u)][k]) for u in us]
                    for x, tag in ((a, "a"), (b, "b")) for k in ("n_ref", "sub", "del", "ins")}
            cp = lambda s, t: (s[f"{t}_sub"] + s[f"{t}_del"] + s[f"{t}_ins"]) / s[f"{t}_n_ref"]  # noqa: E731
            e = of_sums(cols, lambda s: cp(s, "a") - cp(s, "b"), bs["n_resamples"], bs["ci_level"], bs["seed"])
            con.append({"a": a, "b": b, "region": region, "n_files": len(us), "cpwer_a_minus_b": f"{e.point:.6f}",
                        "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    if con:
        with open(out / "contrasts.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(con[0]))
            w.writeheader(); w.writerows(con)
    log.finish(out, meetings_included=included, cpwer_vs_meeteval_max_abs_diff=worst_meeteval)
    print(f"run {log.short_hash}: cpWER matches MeetEval on every file -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
