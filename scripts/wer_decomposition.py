"""Split speaker-attributed WER into transcription, word-rule and diarisation
parts (config comments give the definitions). Our region-split counts are
checked against MeetEval on every file: DI-cpWER and cpWER totals must match
exactly, else the run aborts."""

import argparse
import csv
import sys

from meeteval.wer import cp_word_error_rate
from meeteval.wer import time_constrained_minimum_permutation_word_error_rate as tcp_wer

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import assign
from diarisation.scoring.meeteval_bridge import di_streams, seglst
from diarisation.scoring.wer import RegionLookup, cp_assignment, score_assignment, tokens
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from attribution_asr import asr_words  # noqa: E402

REGIONS = ("all", "single", "overlap")
KEYS = ("n_ref", "sub", "del", "ins")


def systems_of(c):
    """{name: rttm dir} of the diarisation systems of a condition. A config with
    one `pyannote_dir` (the original form) means {"pyannote": that dir}."""
    return dict(c["systems"]) if "systems" in c else {"pyannote": c["pyannote_dir"]}


def totals(acc):
    tot = {k: sum(acc[r][k] for r in acc) for k in KEYS}
    return {"all": tot, "single": acc["single"], "overlap": acc["overlap"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/wer_decomposition_whisper_turbo.yaml")
    ap.add_argument("--conditions", nargs="+")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.conditions:
        cfg["conditions"] = {c: v for c, v in cfg["conditions"].items() if c in args.conditions}
    ami, wcfg, bs = load_yaml(cfg["dataset"]), load_yaml(cfg["words"]), cfg["bootstrap"]
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"],
                 packages=("numpy", "scipy", "rapidfuzz", "meeteval", "whisper-normalizer"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part = cfg["partition"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()
    if cfg.get("uris"):            # optional subset (e.g. the large-v3 check)
        uris = [u for u in uris if u in cfg["uris"]]

    rows, included, di_start_by = [], {}, {}
    for cond, c in cfg["conditions"].items():
        included[cond] = []
        for uri in uris:
            asr_p = REPO_ROOT / c["asr_dir"] / f"{uri}.words.tsv"
            if not asr_p.exists():
                continue
            included[cond].append(uri)
            ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
            uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
            region = RegionLookup(ref, uem)
            ref_streams = {}
            for w in read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv"):
                ref_streams.setdefault(w.speaker, []).append((w.text, w.start, w.end))
            ref_streams = {k: tokens(v) for k, v in ref_streams.items()}
            words = asr_words(asr_p)
            ref_sl = seglst(ref_streams, uri)

            # label the words with the word rule: perfect diarisation, then each system
            labelled = {}
            runs = [("oracle", REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part)] + \
                [(name, REPO_ROOT / d) for name, d in systems_of(c).items()]
            for run, hyp_dir in runs:
                hyp = read_rttm(hyp_dir / f"{uri}.rttm")[uri]
                hs = {}
                for w, lab in zip(words, assign(words, hyp)):
                    hs.setdefault(lab if lab is not None else "__unassigned__", []).append((w.text, w.start, w.end))
                labelled[run] = {k: tokens(v) for k, v in hs.items()}

            # transcription only: DI-cpWER from every labelling, lowest kept
            di_s, di, di_start = di_streams(ref_streams, [(r, labelled[r]) for r, _ in runs], uri)
            acc = score_assignment(ref_streams, di_s, {r: r for r in ref_streams}, region)
            ours = sum(a["sub"] + a["del"] + a["ins"] for a in acc.values())
            if ours != di.errors:
                print(f"ABORT {cond} {uri}: DI-cpWER ours {ours} != MeetEval {di.errors}", file=sys.stderr)
                return 2
            di_start_by[(cond, uri)] = di_start
            measures = {"di_cpwer": totals(acc)}

            for run, _ in runs:
                hyp_streams = labelled[run]
                acc = score_assignment(ref_streams, hyp_streams, cp_assignment(ref_streams, hyp_streams), region)
                me = cp_word_error_rate(ref_sl, seglst(hyp_streams, uri))
                if sum(a["sub"] + a["del"] + a["ins"] for a in acc.values()) != me.errors:
                    print(f"ABORT {cond} {uri} {run}: cpWER differs from MeetEval", file=sys.stderr)
                    return 2
                measures[f"cpwer_{run}"] = totals(acc)
                t = tcp_wer(ref_sl, seglst(hyp_streams, uri), collar=cfg["tcp_collar"])
                measures[f"tcpwer_{run}"] = {"all": {"n_ref": t.length, "sub": t.substitutions,
                                                     "del": t.deletions, "ins": t.insertions}}
            for m, by in measures.items():
                for reg, a in by.items():
                    rows.append({"condition": cond, "uri": uri, "measure": m, "region": reg, **{k: a[k] for k in KEYS}})
            print(f"  {cond} {uri}: DI {measures['di_cpwer']['all']}", flush=True)

    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---------------- everything below reads only per_file.csv ----------------
    back = list(csv.DictReader(open(out / "per_file.csv")))
    idx = {(r["condition"], r["measure"], r["region"], r["uri"]): r for r in back}
    agg = []
    for cond in cfg["conditions"]:
        us = included[cond]
        if not us:
            continue
        systems = list(systems_of(cfg["conditions"][cond]))
        for reg in REGIONS:
            runs = ["oracle"] + systems
            ms = ["di_cpwer"] + [f"cpwer_{r}" for r in runs] + ([f"tcpwer_{r}" for r in runs] if reg == "all" else [])
            cols = {f"{m}|{k}": [float(idx[(cond, m, reg, u)][k]) for u in us] for m in ms for k in KEYS}
            wer = lambda s, m: (s[f"{m}|sub"] + s[f"{m}|del"] + s[f"{m}|ins"]) / s[f"{m}|n_ref"]  # noqa: E731
            stats = {m: (lambda m: lambda s: wer(s, m))(m) for m in ms}
            stats["part_transcription"] = lambda s: wer(s, "di_cpwer")
            stats["part_word_rule"] = lambda s: wer(s, "cpwer_oracle") - wer(s, "di_cpwer")
            # one system: "part_diarisation" as before; several: one per system,
            # and each later system's cpWER minus the first's (paired over meetings)
            for sy in systems:
                name = "part_diarisation" if len(systems) == 1 else f"part_diarisation_{sy}"
                stats[name] = (lambda sy: lambda s: wer(s, f"cpwer_{sy}") - wer(s, "cpwer_oracle"))(sy)
            for sy in systems[1:]:
                stats[f"cpwer_{sy}_minus_{systems[0]}"] = (
                    lambda sy: lambda s: wer(s, f"cpwer_{sy}") - wer(s, f"cpwer_{systems[0]}"))(sy)
            stats["di_del"] = lambda s: s["di_cpwer|del"] / s["di_cpwer|n_ref"]
            if reg == "all":
                for r in runs:
                    stats[f"tcp_minus_cp_{r}"] = (lambda r: lambda s: wer(s, f"tcpwer_{r}") - wer(s, f"cpwer_{r}"))(r)
            for name, f in stats.items():
                e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
                agg.append({"condition": cond, "region": reg, "statistic": name, "n_files": len(us),
                            "n_ref_tokens": int(sum(cols["di_cpwer|n_ref"])),
                            "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)

    # Optional paired contrasts between two conditions on the same meetings
    # (e.g. the same words with two timings): statistic(b) - statistic(a),
    # bootstrapped over meetings, and the number of meetings where it is < 0.
    con = []
    for a, b in cfg.get("contrasts", []):
        us = [u for u in included[a] if u in included[b]]
        sy_a, sy_b = list(systems_of(cfg["conditions"][a])), list(systems_of(cfg["conditions"][b]))
        if sy_a != sy_b:
            print(f"ABORT contrast {a} vs {b}: different systems", file=sys.stderr)
            return 2
        for reg in REGIONS:
            ms = ["di_cpwer"] + [f"cpwer_{r}" for r in ["oracle"] + sy_a]
            cols = {f"{c}|{m}|{k}": [float(idx[(c, m, reg, u)][k]) for u in us]
                    for c in (a, b) for m in ms for k in KEYS}
            wer = lambda s, c, m: (s[f"{c}|{m}|sub"] + s[f"{c}|{m}|del"] + s[f"{c}|{m}|ins"]) / s[f"{c}|{m}|n_ref"]  # noqa: E731
            parts = {m: (lambda m: lambda s, c: wer(s, c, m))(m) for m in ms}
            parts["part_word_rule"] = lambda s, c: wer(s, c, "cpwer_oracle") - wer(s, c, "di_cpwer")
            for sy in sy_a:
                name = "part_diarisation" if len(sy_a) == 1 else f"part_diarisation_{sy}"
                parts[name] = (lambda sy: lambda s, c: wer(s, c, f"cpwer_{sy}") - wer(s, c, "cpwer_oracle"))(sy)
            for name, f in parts.items():
                d = lambda s, f=f: f(s, b) - f(s, a)  # noqa: E731
                e = of_sums(cols, d, bs["n_resamples"], bs["ci_level"], bs["seed"])
                per = [d({k: v[i] for k, v in cols.items()}) for i in range(len(us))]
                con.append({"a": a, "b": b, "region": reg, "statistic": f"{name}_b_minus_a", "n_files": len(us),
                            "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}",
                            "n_files_below_zero": sum(x < 0 for x in per)})
    if con:
        with open(out / "contrasts.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(con[0]))
            w.writeheader(); w.writerows(con)
    log.finish(out, meetings_included=included,
               di_cpwer_best_start={f"{c}/{u}": r for (c, u), r in di_start_by.items()})
    print(f"run {log.short_hash}: DI-cpWER and cpWER match MeetEval on every file -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
