"""Re-time Whisper words with WhisperX forced alignment (config comments).

Runs in venv-align. Resumable: meetings with an existing output are skipped.
Words keep their text and order; a word WhisperX cannot time (no alignable
characters, e.g. digits) keeps its original timing and is counted.

    python scripts/align_whisperx.py --config configs/systems/whisperx_align.yaml
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


def segments(words, max_gap, max_len):
    out, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > max_gap or w["end"] - cur[0]["start"] > max_len):
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/systems/whisperx_align.yaml")
    ap.add_argument("--uris", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / args.config))
    ds = yaml.safe_load(open(REPO / cfg["dataset"]))
    paths = yaml.safe_load(open(REPO / "configs" / "paths.yaml"))
    audio_root = Path(paths["audio_root"]).expanduser() / "amicorpus"

    import torch
    import whisperx
    torch.set_num_threads(cfg["cpu_threads"])
    model, meta = whisperx.load_align_model(language_code=cfg["language"], device=cfg["device"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

    for cond in cfg["conditions"]:
        stream = ds["conditions"][cond]
        for part in cfg["partitions"]:
            src = REPO / cfg["input_dir"] / cond / part
            out = REPO / cfg["out_dir"] / cond / part
            out.mkdir(parents=True, exist_ok=True)
            uris = (REPO / ds["out_dir"] / "lists" / f"{part}.txt").read_text().split()
            uris = [u for u in uris if not args.uris or u in args.uris]
            timing_p = out / "timing.csv"
            timing = list(csv.DictReader(open(timing_p))) if timing_p.exists() else []
            for uri in uris:
                tsv = out / f"{uri}.words.tsv"
                if tsv.exists() and not args.force:
                    continue
                words = [{"start": float(r["start"]), "end": float(r["end"]), "word": r["word"], "probability": r["probability"]}
                         for r in csv.DictReader(open(src / f"{uri}.words.tsv", encoding="utf-8"), delimiter="\t")]
                segs = segments(words, cfg["segment_max_gap"], cfg["segment_max_len"])
                wav = audio_root / uri / "audio" / f"{uri}.{stream}.wav"
                audio = whisperx.load_audio(str(wav))
                t0 = time.perf_counter()
                res = whisperx.align([{"text": " ".join(w["word"] for w in s), "start": s[0]["start"], "end": s[-1]["end"]}
                                      for s in segs], model, meta, audio, cfg["device"],
                                     interpolate_method=cfg["interpolate_method"], return_char_alignments=False)
                wall = time.perf_counter() - t0
                aligned = [w for seg in res["segments"] for w in seg.get("words", [])]
                if len(aligned) != len(words) or any(a["word"] != w["word"] for a, w in zip(aligned, words)):
                    print(f"ABORT {uri}: aligned words ({len(aligned)}) differ from input ({len(words)})", file=sys.stderr)
                    return 2
                kept = 0
                rows = []
                for a, w in zip(aligned, words):
                    s, e = a.get("start"), a.get("end")
                    if s is None or e is None or not (e >= s):
                        s, e, kept = w["start"], w["end"], kept + 1
                    rows.append((s, e, w["word"], w["probability"]))
                tmp = tsv.with_suffix(".tmp")
                with open(tmp, "w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    wr.writerow(["start", "end", "word", "probability"])
                    for s, e, wd, p in rows:
                        wr.writerow([f"{s:.3f}", f"{e:.3f}", wd, p])
                tmp.replace(tsv)
                dur = len(audio) / 16000
                timing = [r for r in timing if r["uri"] != uri] + [{"uri": uri, "audio_s": f"{dur:.3f}", "wall_s": f"{wall:.1f}",
                          "rtf": f"{wall / dur:.4f}", "n_words": len(words), "n_segments": len(segs), "n_kept_original_time": kept}]
                with open(timing_p, "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=list(timing[0]))
                    wr.writeheader(); wr.writerows(timing)
                print(f"[{cond}/{part}] {uri}: {len(words)} words in {len(segs)} segments, {wall:.0f} s "
                      f"(RTF {wall / dur:.3f}), {kept} kept original time", flush=True)
            (out / "run.json").write_text(json.dumps({
                "experiment": f"align_{cfg['system']}", "config": cfg, "condition": cond, "partition": part,
                "git_commit": commit, "git_dirty": dirty, "n_files": len(timing),
                "versions": {"python": platform.python_version(),
                             **{p: metadata.version(p) for p in ("whisperx", "torch", "torchaudio", "transformers", "nltk")}},
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_seconds_total": round(sum(float(r["wall_s"]) for r in timing), 1)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
