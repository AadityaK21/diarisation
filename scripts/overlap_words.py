"""Which overlapped words are lost, and to what? (config comments define the
features and outcomes). Writes the per-unit table, per-meeting counts per
group, and group rates with bootstrap CIs over meetings (from the counts)."""

import argparse
import bisect
import csv
import math
import sys

import numpy as np
import soundfile as sf

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami_words import ensure_archive, headset_channels, read_tsv
from diarisation.intervals import per_speaker_union
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring.attribution import assign
from diarisation.scoring.meeteval_bridge import di_streams, ref_token_ops
from diarisation.scoring.wer import RegionLookup, Token, tokens
from diarisation.stats.bootstrap import of_sums

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from attribution_asr import asr_words  # noqa: E402


def region_of(regions, t):
    """(start, end) of the region in a sorted list containing t, else None."""
    i = bisect.bisect_right(regions, (t, math.inf)) - 1
    if i >= 0 and regions[i][0] <= t < regions[i][1]:
        return regions[i]
    return None


class Energy:
    """Mean square of a headset channel over any interval, from a cumulative sum."""

    def __init__(self, path):
        x, self.sr = sf.read(str(path), dtype="int16")
        x = x.astype(np.float64)
        self.c = np.concatenate([[0.0], np.cumsum(x * x)])

    def __call__(self, a, b):
        i, j = int(round(a * self.sr)), int(round(b * self.sr))
        j = min(j, len(self.c) - 1)
        return (self.c[j] - self.c[i]) / (j - i) if j > i else float("nan")


def describe(units, by_spk, turn_words, energies, min_dur):
    """Add role / turn_words / level_db to overlapped units (dicts with speaker, start, end)."""
    for u in units:
        mid = 0.5 * (u["start"] + u["end"])
        own = region_of(by_spk[u["speaker"]], mid)
        others = [(s, region_of(r, mid)) for s, r in by_spk.items() if s != u["speaker"]]
        others = [(s, reg) for s, reg in others if reg is not None]
        own_start = own[0] if own else u["start"]
        u["role"] = "first" if all(own_start < reg[0] for _, reg in others) else "second"
        u["turn_words"] = turn_words.get((u["speaker"], own), 1)
        lvl = float("nan")
        if energies and u["end"] - u["start"] >= min_dur and u["speaker"] in energies \
                and all(s in energies for s, _ in others) and others:
            e_own = energies[u["speaker"]](u["start"], u["end"])
            e_oth = sum(energies[s](u["start"], u["end"]) for s, _ in others)
            if e_own > 0 and e_oth > 0:
                lvl = 10 * math.log10(e_own / e_oth)
        u["level_db"] = lvl


def level_bin(x, edges):
    if x != x:
        return "unknown"
    i = bisect.bisect_right(edges, x)
    lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
    hi = "inf" if i == len(edges) else f"{edges[i]:g}"
    return f"[{lo},{hi})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/overlap_words_whisper_turbo.yaml")
    ap.add_argument("--conditions", nargs="+")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.conditions:
        cfg["conditions"] = {c: v for c, v in cfg["conditions"].items() if c in args.conditions}
    ami, wcfg, bs = load_yaml(cfg["dataset"]), load_yaml(cfg["words"]), cfg["bootstrap"]
    paths = load_paths()
    chans = headset_channels(ensure_archive(wcfg, paths["cache_root"]))
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=bs["seed"],
                 packages=("numpy", "scipy", "soundfile", "meeteval", "rapidfuzz", "whisper-normalizer"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    part, edges = cfg["partition"], cfg["level_edges_db"]
    uris = (REPO_ROOT / ami["out_dir"] / "lists" / f"{part}.txt").read_text().split()
    audio = paths["audio_root"] / "amicorpus"
    short = cfg["short_turn_max_words"]

    units_rows, counts, missing_audio = [], [], []
    for uri in uris:
        ref = read_rttm(REPO_ROOT / ami["out_dir"] / cfg["annotation"] / "rttm" / part / f"{uri}.rttm")[uri]
        uem = read_uem(REPO_ROOT / ami["out_dir"] / "uem" / part / f"{uri}.uem")[uri]
        region = RegionLookup(ref, uem)
        by_spk = per_speaker_union(ref)
        words = read_tsv(REPO_ROOT / wcfg["out_dir"] / part / f"{uri}.tsv")
        turn_words = {}
        for w in words:
            reg = region_of(by_spk.get(w.speaker, []), 0.5 * (w.start + w.end))
            turn_words[(w.speaker, reg)] = turn_words.get((w.speaker, reg), 0) + 1
        energies = {}
        for spk, ch in chans[uri].items():
            p = audio / uri / "audio" / f"{uri}.Headset-{ch}.wav"
            if p.exists():
                energies[spk] = Energy(p)
            else:
                missing_audio.append(str(p))

        # attribution with perfect diarisation, per overlapped reference word
        labels = assign(words, ref)
        attr_units = []
        for w, lab in zip(words, labels):
            if region(Token(w.text, w.start, w.end)) == "overlap":
                attr_units.append({"analysis": "attribution", "condition": "-", "uri": uri, "speaker": w.speaker,
                                   "start": w.start, "end": w.end, "text": w.text,
                                   "outcome": "correct" if lab == w.speaker else "other_talker"})
        describe(attr_units, by_spk, turn_words, energies, cfg["min_word_seconds"])
        units = list(attr_units)

        # transcription outcome per overlapped reference token, per condition
        ref_streams = {}
        for w in words:
            ref_streams.setdefault(w.speaker, []).append((w.text, w.start, w.end))
        ref_streams = {k: tokens(v) for k, v in ref_streams.items()}
        for cond, c in cfg["conditions"].items():
            asr_p = REPO_ROOT / c["asr_dir"] / f"{uri}.words.tsv"
            if not asr_p.exists():
                continue
            hw = asr_words(asr_p)
            labelled = []
            for run, hyp in (("oracle", ref), ("pyannote", read_rttm(REPO_ROOT / c["pyannote_dir"] / f"{uri}.rttm")[uri])):
                hs = {}
                for w, lab in zip(hw, assign(hw, hyp)):
                    hs.setdefault(lab if lab is not None else "__unassigned__", []).append((w.text, w.start, w.end))
                labelled.append((run, {k: tokens(v) for k, v in hs.items()}))
            di_s, _, _ = di_streams(ref_streams, labelled, uri)
            asr_units = []
            for spk, toks in ref_streams.items():
                for t, op in zip(toks, ref_token_ops(toks, di_s.get(spk, []))):
                    if region(t) == "overlap":
                        asr_units.append({"analysis": "asr", "condition": cond, "uri": uri, "speaker": spk,
                                          "start": t.start, "end": t.end, "text": t.text,
                                          "outcome": {"ok": "correct", "sub": "substituted", "del": "deleted"}[op]})
            describe(asr_units, by_spk, turn_words, energies, cfg["min_word_seconds"])
            units += asr_units

        for u in units:
            u["length"] = f"1-{short} words" if u["turn_words"] <= short else f">{short} words"
            u["level"] = level_bin(u["level_db"], edges)
            units_rows.append(u)
        # per-meeting counts per group
        for (ana, cond) in sorted({(u["analysis"], u["condition"]) for u in units}):
            sel = [u for u in units if u["analysis"] == ana and u["condition"] == cond]
            bad = "other_talker" if ana == "attribution" else "deleted"
            for gtype in ("all", "role", "length", "level"):
                groups = {}
                for u in sel:
                    g = "all" if gtype == "all" else u[gtype]
                    n, b, s = groups.get(g, (0, 0, 0))
                    groups[g] = (n + 1, b + (u["outcome"] == bad), s + (u["outcome"] == "substituted"))
                for g, (n, b, s) in groups.items():
                    counts.append({"analysis": ana, "condition": cond, "uri": uri, "group_type": gtype, "group": g,
                                   "n": n, "n_bad": b, "n_sub": s})
        print(f"  {uri}: {len(attr_units)} overlapped words, {len(units) - len(attr_units)} ASR tokens, "
              f"{len(energies)} headsets", flush=True)

    with open(out / "units.csv", "w", newline="") as fh:
        cols = ["analysis", "condition", "uri", "speaker", "start", "end", "text", "outcome", "role", "turn_words",
                "length", "level_db", "level"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for u in units_rows:
            w.writerow({**u, "start": f"{u['start']:.3f}", "end": f"{u['end']:.3f}",
                        "level_db": "" if u["level_db"] != u["level_db"] else f"{u['level_db']:.2f}"})
    with open(out / "per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(counts[0]))
        w.writeheader(); w.writerows(counts)

    # ---------------- aggregates read only per_file.csv ----------------
    back = list(csv.DictReader(open(out / "per_file.csv")))
    agg = []
    for key in sorted({(r["analysis"], r["condition"], r["group_type"], r["group"]) for r in back}):
        ana, cond, gtype, g = key
        us = sorted({r["uri"] for r in back if (r["analysis"], r["condition"]) == (ana, cond)})
        idx = {r["uri"]: r for r in back if (r["analysis"], r["condition"], r["group_type"], r["group"]) == key}
        cols = {k: [float(idx[u][k]) if u in idx else 0.0 for u in us] for k in ("n", "n_bad", "n_sub")}
        if sum(cols["n"]) == 0:
            continue
        e = of_sums(cols, lambda s: s["n_bad"] / s["n"], bs["n_resamples"], bs["ci_level"], bs["seed"])
        agg.append({"analysis": ana, "condition": cond, "group_type": gtype, "group": g, "n_files": len(us),
                    "n_with_group": len(idx), "n_units": int(sum(cols["n"])), "share_of_units": "",
                    "rate_bad": f"{e.point:.6f}", "lo": f"{e.lo:.6f}", "hi": f"{e.hi:.6f}",
                    "rate_sub": f"{sum(cols['n_sub']) / sum(cols['n']):.6f}"})
    tot = {(a["analysis"], a["condition"]): a["n_units"] for a in agg if a["group_type"] == "all"}
    for a in agg:
        a["share_of_units"] = f"{a['n_units'] / tot[(a['analysis'], a['condition'])]:.6f}"
    with open(out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    log.finish(out, missing_audio=missing_audio)
    print(f"run {log.short_hash} -> {cfg['out_dir']}/ ({len(missing_audio)} headset files missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
