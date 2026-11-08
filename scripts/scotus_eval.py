"""Supreme Court set: DER (checked against md-eval), how system speakers map
to people, and speaker-attributed WER under cpWER, the brief's fixed
many-to-one mapping, a fixed one-to-one mapping and DI-cpWER (MeetEval)."""

import argparse
import csv
import sys
from collections import defaultdict

from meeteval.wer import cp_word_error_rate

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.data.ami_words import read_tsv
from diarisation.intervals import per_speaker_union
from diarisation.io.rttm import read_rttm
from diarisation.io.uem import read_uem
from diarisation.runlog import RunLog
from diarisation.scoring import mdeval
from diarisation.scoring.attribution import assign, mappings
from diarisation.scoring.der import overlap_seconds, score
from diarisation.scoring.meeteval_bridge import di_streams, seglst
from diarisation.scoring.wer import RegionLookup, cp_assignment, merge_by_mapping, score_assignment, tokens

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from attribution_asr import asr_words  # noqa: E402


def errs(acc):
    return sum(a["sub"] + a["del"] + a["ins"] for a in acc.values()), sum(a["n_ref"] for a in acc.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/scotus_pyannote31_whisper.yaml")
    cfg = load_yaml(ap.parse_args().config)
    ds = load_yaml(cfg["dataset"])
    log = RunLog(cfg["experiment"], {"experiment": cfg}, seed=None,
                 packages=("numpy", "scipy", "rapidfuzz", "meeteval", "whisper-normalizer"))
    out = REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    root = REPO_ROOT / ds["out_dir"]
    uris = (root / "lists" / f"{cfg['partition']}.txt").read_text().split()
    summary, speakers, clusters = [], [], []
    for uri in uris:
        ref_p, uem_p, hyp_p = root / "rttm" / f"{uri}.rttm", root / "uem" / f"{uri}.uem", REPO_ROOT / cfg["hyp_dir"] / f"{uri}.rttm"
        ref, uem, hyp = read_rttm(ref_p)[uri], read_uem(uem_p)[uri], read_rttm(hyp_p)[uri]
        row = lambda k, v: summary.append({"uri": uri, "statistic": k, "value": v})  # noqa: E731
        row("n_reference_speakers", len({s.speaker for s in ref}))
        row("n_system_speakers", len({s.speaker for s in hyp}))
        for collar in cfg["collars"]:
            ours = score(ref, hyp, uem, collar, False, cfg["frame_step"])
            md = mdeval.run(ref_p, hyp_p, uem_p, collar, False)
            for k in ("scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"):
                if abs(getattr(ours, k) - getattr(md, k)) > 0.001:
                    print(f"ABORT: {k} ours {getattr(ours, k)} vs md-eval {getattr(md, k)}", file=sys.stderr)
                    return 2
            c = f"collar{collar:g}"
            row(f"der_{c}", f"{ours.der:.6f}")
            for k in ("missed_s", "false_alarm_s", "confusion_s", "scored_speaker_s"):
                row(f"{k}_{c}", f"{getattr(ours, k):.3f}")

        # how system speakers relate to people
        maps = mappings(ref, hyp, uem, cfg["frame_step"])
        ov = overlap_seconds(ref, hyp, uem, cfg["frame_step"], "all")
        talk = {s: sum(e - b for b, e in r) for s, r in per_speaker_union(ref).items()}
        htime = {s: sum(e - b for b, e in r) for s, r in per_speaker_union(hyp).items()}
        for h in sorted(htime):
            by_ref = sorted(((sec, r) for (r, hh), sec in ov.items() if hh == h and sec > 0), reverse=True)
            covered = [r for sec, r in by_ref if sec >= cfg["cluster_share_min"] * htime[h]]
            clusters.append({"uri": uri, "system_speaker": h, "seconds": f"{htime[h]:.1f}",
                             "top_person": by_ref[0][1] if by_ref else "", "top_share": f"{by_ref[0][0] / htime[h]:.3f}" if by_ref else "",
                             "people_covered": ";".join(covered), "majority_map": maps["majority_single"].get(h, ""),
                             "optimal_map": maps["optimal"].get(h, ""), "onetoone_single_map": maps["onetoone_single"].get(h, "")})
        for r in sorted(talk, key=lambda r: -talk[r]):
            speakers.append({"uri": uri, "person": r, "talk_seconds": f"{talk[r]:.1f}",
                             "majority_system_speakers": ";".join(sorted(h for h, rr in maps["majority_single"].items() if rr == r)),
                             "optimal_system_speaker": next((h for h, rr in maps["optimal"].items() if rr == r), "")})
        merged = sum(1 for c in clusters if c["uri"] == uri and len(c["people_covered"].split(";")) > 1)
        row("system_speakers_covering_2plus_people", merged)
        row("people_without_own_system_speaker_optimal", len(talk) - len(maps["optimal"]))

        # speaker-attributed WER
        words = read_tsv(root / "words" / f"{uri}.tsv")
        ref_streams = defaultdict(list)
        for w in words:
            ref_streams[w.speaker].append((w.text, w.start, w.end))
        ref_streams = {k: tokens(v) for k, v in ref_streams.items()}
        region = RegionLookup(ref, uem)
        hw = asr_words(REPO_ROOT / cfg["asr_dir"] / f"{uri}.words.tsv")
        labelled = {}
        for run, segs in (("oracle", ref), ("pyannote", hyp)):
            hs = defaultdict(list)
            for w, lab in zip(hw, assign(hw, segs)):
                hs[lab if lab is not None else "__unassigned__"].append((w.text, w.start, w.end))
            labelled[run] = {k: tokens(v) for k, v in hs.items()}
        ref_sl = seglst(ref_streams, uri)
        for run in ("oracle", "pyannote"):
            hs = labelled[run]
            acc = score_assignment(ref_streams, hs, cp_assignment(ref_streams, hs), region)
            e, n = errs(acc)
            me = cp_word_error_rate(ref_sl, seglst(hs, uri))
            if (e, n) != (me.errors, me.length):
                print(f"ABORT: cpWER {run} ours {(e, n)} vs MeetEval {(me.errors, me.length)}", file=sys.stderr)
                return 2
            row(f"cpwer_{run}", f"{e / n:.6f}")
            sys_map = mappings(ref, ref if run == "oracle" else hyp, uem, cfg["frame_step"])
            merged_streams = merge_by_mapping(hs, sys_map["majority_single"])
            e2, _ = errs(score_assignment(ref_streams, merged_streams, {r: f"ref:{r}" for r in ref_streams}, region))
            row(f"fixed_manytoone_wer_{run}", f"{e2 / n:.6f}")
            one = {r: h for h, r in sys_map["onetoone_single"].items()}
            e3, _ = errs(score_assignment(ref_streams, hs, {r: one.get(r) for r in ref_streams}, region))
            if e3 < e:
                print("ABORT: one-to-one fixed mapping beats cpWER", file=sys.stderr)
                return 3
            row(f"fixed_onetoone_wer_{run}", f"{e3 / n:.6f}")
        di_s, di, start = di_streams(ref_streams, [("oracle", labelled["oracle"]), ("pyannote", labelled["pyannote"])], uri)
        e4, n4 = errs(score_assignment(ref_streams, di_s, {r: r for r in ref_streams}, region))
        if e4 != di.errors:
            print("ABORT: DI-cpWER differs from MeetEval", file=sys.stderr)
            return 2
        row("di_cpwer", f"{e4 / n4:.6f}")
        row("n_reference_tokens", n4)
        row("n_output_words", len(hw))

    for name, data in (("summary.csv", summary), ("speakers.csv", speakers), ("clusters.csv", clusters)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    log.finish(out)
    for r in summary:
        print(f"  {r['statistic']:42s} {r['value']}")
    print(f"run {log.short_hash} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
