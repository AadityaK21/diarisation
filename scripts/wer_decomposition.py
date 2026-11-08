"""Split speaker-attributed WER into transcription, word-rule and diarisation
parts (config comments give the definitions). Our region-split counts are
checked against MeetEval on every file: DI-cpWER and cpWER totals must match
exactly, else the run aborts."""

import argparse
import csv
import sys

from meeteval.io.seglst import SegLST
from meeteval.wer import cp_word_error_rate, greedy_di_cp_word_error_rate
from meeteval.wer import time_constrained_minimum_permutation_word_error_rate as tcp_wer

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import assign
from diarisation.scoring.wer import RegionLookup, Token, cp_assignment, score_assignment, tokens
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from attribution_asr import asr_words  # noqa: E402

REGIONS = ("all", "single", "overlap")
KEYS = ("n_ref", "sub", "del", "ins")


def seglst(streams, uri):
    """{speaker: [Token]} -> SegLST with one segment per token."""
    return SegLST([{"session_id": uri, "speaker": spk, "start_time": t.start, "end_time": t.end, "words": t.text}
                   for spk, toks in streams.items() for t in toks])


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

            # label the words two ways (perfect diarisation, pyannote) with the word rule
            labelled = {}
            for run, hyp_dir in (("oracle", REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part),
                                 ("pyannote", REPO_ROOT / c["pyannote_dir"])):
                hyp = read_rttm(hyp_dir / f"{uri}.rttm")[uri]
                hs = {}
                for w, lab in zip(words, assign(words, hyp)):
                    hs.setdefault(lab if lab is not None else "__unassigned__", []).append((w.text, w.start, w.end))
                labelled[run] = {k: tokens(v) for k, v in hs.items()}

            # transcription only: DI-cpWER. MeetEval's greedy search starts from
            # the cpWER assignment of the hypothesis labels, so run it from both
            # labellings and keep the lower (both are upper bounds on DI-cpWER;
            # starting from the oracle labels guarantees DI <= cpWER_oracle).
            best = None
            for run in ("oracle", "pyannote"):
                hyp_sl = seglst(labelled[run], uri)
                di = greedy_di_cp_word_error_rate(ref_sl, hyp_sl)
                if best is None or di.errors < best[0].errors:
                    best = (di, hyp_sl, run)
            di, hyp_sl, di_start = best
            _, reassigned = di.apply_assignment(ref_sl, hyp_sl)
            di_streams = {}
            for s in reassigned:
                di_streams.setdefault(s["speaker"], []).append(Token(s["words"], s["start_time"], s["end_time"]))
            # MeetEval orders each stream by time; tokens of one word share their
            # times and keep their order (stable sort)
            di_streams = {k: sorted(v, key=lambda t: (t.start, t.end)) for k, v in di_streams.items()}
            acc = score_assignment(ref_streams, di_streams, {r: r for r in ref_streams}, region)
            ours = sum(a["sub"] + a["del"] + a["ins"] for a in acc.values())
            if ours != di.errors:
                print(f"ABORT {cond} {uri}: DI-cpWER ours {ours} != MeetEval {di.errors}", file=sys.stderr)
                return 2
            di_start_by[(cond, uri)] = di_start
            measures = {"di_cpwer": totals(acc)}

            for run in ("oracle", "pyannote"):
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
        for reg in REGIONS:
            ms = ["di_cpwer", "cpwer_oracle", "cpwer_pyannote"] + (["tcpwer_oracle", "tcpwer_pyannote"] if reg == "all" else [])
            cols = {f"{m}|{k}": [float(idx[(cond, m, reg, u)][k]) for u in us] for m in ms for k in KEYS}
            wer = lambda s, m: (s[f"{m}|sub"] + s[f"{m}|del"] + s[f"{m}|ins"]) / s[f"{m}|n_ref"]  # noqa: E731
            stats = {m: (lambda m: lambda s: wer(s, m))(m) for m in ms}
            stats["part_transcription"] = lambda s: wer(s, "di_cpwer")
            stats["part_word_rule"] = lambda s: wer(s, "cpwer_oracle") - wer(s, "di_cpwer")
            stats["part_diarisation"] = lambda s: wer(s, "cpwer_pyannote") - wer(s, "cpwer_oracle")
            stats["di_del"] = lambda s: s["di_cpwer|del"] / s["di_cpwer|n_ref"]
            if reg == "all":
                stats["tcp_minus_cp_oracle"] = lambda s: wer(s, "tcpwer_oracle") - wer(s, "cpwer_oracle")
                stats["tcp_minus_cp_pyannote"] = lambda s: wer(s, "tcpwer_pyannote") - wer(s, "cpwer_pyannote")
            for name, f in stats.items():
                e = of_sums(cols, f, bs["n_resamples"], bs["ci_level"], bs["seed"])
                agg.append({"condition": cond, "region": reg, "statistic": name, "n_files": len(us),
                            "n_ref_tokens": int(sum(cols["di_cpwer|n_ref"])),
                            "point": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}"})
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    log.finish(out, meetings_included=included,
               di_cpwer_best_start={f"{c}/{u}": r for (c, u), r in di_start_by.items()})
    print(f"run {log.short_hash}: DI-cpWER and cpWER match MeetEval on every file -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
