"""Step 6b: transcribe AMI meetings with faster-whisper, one word TSV per meeting.

Runs in its own environment (faster-whisper only). Resumable: meetings with
an existing TSV are skipped unless --force.

    python scripts/transcribe.py --config configs/systems/whisper_turbo_cpu.yaml --conditions ihm_mix
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def longest_repeat(words) -> int:
    """Longest run of the same word in a row: a cheap check for decoder loops."""
    best = cur = 0
    prev = None
    for w in words:
        cur = cur + 1 if w == prev else 1
        best, prev = max(best, cur), w
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+")
    ap.add_argument("--dataset", help="override the system config's dataset (e.g. configs/data/scotus_oyez.yaml)")
    ap.add_argument("--partitions", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--uris", nargs="+", help="only these meetings (e.g. to check one file)")
    ap.add_argument("--out-dir", help="override cfg out_dir (e.g. a scratch check)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / args.config))
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.partitions:
        cfg["partitions"] = args.partitions
    ds = yaml.safe_load(open(REPO / cfg["dataset"]))
    paths = yaml.safe_load(open(REPO / "configs" / "paths.yaml"))
    audio_root = Path(paths["audio_root"]).expanduser()

    import ctranslate2
    from faster_whisper import WhisperModel
    ctranslate2.set_random_seed(cfg["seed"])  # temperature-fallback sampling
    model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"],
                         cpu_threads=cfg["cpu_threads"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()

    for cond in args.conditions or cfg["conditions"]:
        stream = ds["conditions"][cond]
        local = ds.get("audio_path")   # local files (e.g. the Supreme Court set)
        for part in cfg["partitions"]:
            out = Path(args.out_dir or REPO / cfg["out_dir"]) / cond / part
            out.mkdir(parents=True, exist_ok=True)
            uris = (REPO / ds["out_dir"] / "lists" / f"{part}.txt").read_text().split()[: args.limit]
            uris = [u for u in uris if not args.uris or u in args.uris]
            timing_p = out / "timing.csv"
            timing = list(csv.DictReader(open(timing_p))) if timing_p.exists() else []
            for i, uri in enumerate(uris, 1):
                tsv = out / f"{uri}.words.tsv"
                if tsv.exists() and not args.force:
                    continue
                wav = (audio_root / local.format(uri=uri)) if local else \
                    (audio_root / "amicorpus" / uri / "audio" / f"{uri}.{stream}.wav")
                t0 = time.perf_counter()
                segs, info = model.transcribe(
                    str(wav), language=cfg["language"], beam_size=cfg["beam_size"],
                    temperature=cfg["temperature"], compression_ratio_threshold=cfg["compression_ratio_threshold"],
                    log_prob_threshold=cfg["log_prob_threshold"], no_speech_threshold=cfg["no_speech_threshold"],
                    condition_on_previous_text=cfg["condition_on_previous_text"], vad_filter=cfg["vad_filter"],
                    word_timestamps=cfg["word_timestamps"])
                words = [w for s in segs for w in (s.words or [])]
                run = longest_repeat(w.word.strip() for w in words)
                wall = time.perf_counter() - t0
                tmp = tsv.with_suffix(".tmp")
                with open(tmp, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    w.writerow(["start", "end", "word", "probability"])
                    for x in words:
                        w.writerow([f"{x.start:.3f}", f"{x.end:.3f}", x.word.strip(), f"{x.probability:.4f}"])
                tmp.replace(tsv)
                timing = [r for r in timing if r["uri"] != uri] + [{"uri": uri, "audio_s": f"{info.duration:.3f}",
                          "wall_s": f"{wall:.1f}", "rtf": f"{wall / info.duration:.4f}", "n_words": len(words),
                          "longest_repeat": run}]
                with open(timing_p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(timing[0]))
                    w.writeheader(); w.writerows(timing)
                print(f"[{cond}/{part} {i}/{len(uris)}] {uri} {info.duration/60:.1f} min, {wall:.0f} s, "
                      f"RTF {wall/info.duration:.2f}, {len(words)} words, longest repeat {run}", flush=True)
            (out / "run.json").write_text(json.dumps({
                "experiment": f"asr_{cfg['system']}", "config": cfg, "condition": cond, "partition": part,
                "git_commit": commit, "seed": cfg["seed"], "n_files": len(timing),
                "versions": {"python": platform.python_version(),
                             **{p: metadata.version(p) for p in ("faster-whisper", "ctranslate2")}},
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_seconds_total": round(sum(float(r["wall_s"]) for r in timing), 1)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
