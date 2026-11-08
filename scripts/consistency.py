"""Per-meeting consistency of the headline effects, with an exact sign test."""

import argparse
import csv
import sys
from collections import defaultdict
from math import comb

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.runlog import RunLog


def sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial p for k successes in n trials at p = 0.5."""
    if n == 0:
        return float("nan")
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(path):
    return list(csv.DictReader(open(REPO_ROOT / path)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/consistency.yaml")
    cfg = load_yaml(ap.parse_args().config)
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=None, packages=())
    src = cfg["sources"]
    claims = []   # (claim, subject, {uri: (a, b)}) ; claim holds when a > b

    der = [r for r in load(src["der"]) if float(r["collar"]) == 0.0 and r["exclude_overlap"] == "0"]
    by = defaultdict(dict)
    for r in der:
        by[(r["system"], r["uri"])][r["region"]] = r
    for system in sorted({r["system"] for r in der}):
        pairs, pairs2 = {}, {}
        for (s, uri), regs in by.items():
            if s != system:
                continue
            err = {g: float(v["missed_s"]) + float(v["false_alarm_s"]) + float(v["confusion_s"]) for g, v in regs.items()}
            tim = {g: float(v["scored_speaker_s"]) for g, v in regs.items()}
            pairs[uri] = (err["overlap"] / tim["overlap"], err["single"] / tim["single"])
            pairs2[uri] = (err["overlap"] / err["all"], tim["overlap"] / tim["all"])
        claims.append(("DER in overlap > DER in single-speaker speech", system, pairs))
        claims.append(("overlap share of DER errors > overlap share of scored time", system, pairs2))

    att = load(src["attribution"])
    idx = {(r["system"], r["mapping"], r["region"], r["uri"]): r for r in att}
    err = lambda r: 1 - int(r["correct"]) / int(r["n_words"])  # noqa: E731
    for system in sorted({r["system"] for r in att}):
        uris = sorted({r["uri"] for r in att if r["system"] == system})
        claims.append(("attribution error in overlap > in single-speaker speech", system,
                       {u: (err(idx[(system, "optimal", "overlap", u)]), err(idx[(system, "optimal", "single", u)])) for u in uris}))
        if system != "oracle_diarisation":
            claims.append(("attribution error, optimal mapping > majority (metric B)", system,
                           {u: (err(idx[(system, "optimal", "all", u)]), err(idx[(system, "majority_single", "all", u)])) for u in uris}))

    asr = load(src["asr"])
    idx = {(r["run"], r["assignment"], r["region"], r["uri"]): r for r in asr}
    wer = lambda r: (int(r["sub"]) + int(r["del"]) + int(r["ins"])) / int(r["n_ref"])  # noqa: E731
    runs = sorted({r["run"] for r in asr})
    for run in runs:
        uris = sorted({r["uri"] for r in asr if r["run"] == run})
        claims.append(("cpWER in overlap > outside overlap", run,
                       {u: (wer(idx[(run, "cp", "overlap", u)]), wer(idx[(run, "cp", "single", u)])) for u in uris}))
        if run.startswith("pyannote"):
            claims.append(("cpWER > fixed many-to-one WER (metric B)", run,
                           {u: (wer(idx[(run, "cp", "all", u)]), wer(idx[(run, "fixed", "all", u)])) for u in uris}))
            oracle = run.replace("pyannote31", "oracle_diarisation")
            if oracle in runs:
                claims.append(("cpWER pyannote > cpWER perfect diarisation", run,
                               {u: (wer(idx[(run, "cp", "all", u)]), wer(idx[(oracle, "cp", "all", u)])) for u in uris}))

    dec = load(src["decomposition"])
    idx = {(r["condition"], r["measure"], r["region"], r["uri"]): r for r in dec}
    for cond in sorted({r["condition"] for r in dec}):
        uris = sorted({r["uri"] for r in dec if r["condition"] == cond})
        f = lambda m, g, u: wer(idx[(cond, m, g, u)])  # noqa: E731
        claims.append(("word-assignment part > diarisation part of cpWER", cond,
                       {u: (f("cpwer_oracle", "all", u) - f("di_cpwer", "all", u),
                            f("cpwer_pyannote", "all", u) - f("cpwer_oracle", "all", u)) for u in uris}))
        claims.append(("transcription-only error (DI-cpWER) in overlap > outside", cond,
                       {u: (f("di_cpwer", "overlap", u), f("di_cpwer", "single", u)) for u in uris}))

    ow = load(src["overlap_words"])
    below, above = ("[-inf,-10)", "[-10,-5)", "[-5,0)"), ("[0,5)", "[5,10)", "[10,inf)")
    for cond in sorted({r["condition"] for r in ow if r["analysis"] == "asr"}):
        sel = [r for r in ow if r["analysis"] == "asr" and r["condition"] == cond]
        uris = sorted({r["uri"] for r in sel})
        def rate(u, gtype, groups):
            rr = [r for r in sel if r["uri"] == u and r["group_type"] == gtype and r["group"] in groups]
            n = sum(int(r["n"]) for r in rr)
            return sum(int(r["n_bad"]) for r in rr) / n if n else float("nan")
        claims.append(("Whisper deletes more overlapped words below 0 dB than above", cond,
                       {u: (rate(u, "level", below), rate(u, "level", above)) for u in uris}))
        claims.append(("Whisper deletes more overlapped words of the second talker than the first", cond,
                       {u: (rate(u, "role", ("second",)), rate(u, "role", ("first",))) for u in uris}))
    dom = load(src["dominance"])
    for system in sorted({r["system"] for r in dom}):
        claims.append(("pyannote shows the louder of two talkers more often than the quieter", system,
                       {r["uri"]: (int(r["louder"]), int(r["quieter"])) for r in dom if r["system"] == system}))

    rows = []
    for claim, subject, pairs in claims:
        holds = sum(a > b for a, b in pairs.values())
        ties = sum(a == b for a, b in pairs.values())
        n = len(pairs) - ties
        rows.append({"claim": claim, "subject": subject, "n_meetings": len(pairs), "n_holds": holds, "n_ties": ties,
                     "p_sign_test": f"{sign_test(holds, n):.3g}",
                     "fails_in": ";".join(sorted(u for u, (a, b) in pairs.items() if a < b))})
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "claims.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    log.finish(out, n_claims=len(rows))
    for r in rows:
        print(f"{r['n_holds']:>2}/{r['n_meetings']} (ties {r['n_ties']}, p={r['p_sign_test']:>8}) {r['subject']:28s} {r['claim']}"
              + (f"  fails: {r['fails_in']}" if r["fails_in"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
