"""Step 5: run a diarisation system over AMI and write one RTTM per meeting.

Resumable: meetings whose RTTM already exists are skipped unless --force.
Per-meeting timing goes to <out>/<condition>/<partition>/timing.csv and the
run record to run.json beside it.

    python scripts/diarise.py --config configs/systems/pyannote31.yaml
    python scripts/diarise.py --config ... --conditions sdm --limit 1   # smoke test
"""

import argparse
import csv
import random
import sys
import time

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami import audio_urls
from diarisation.io.rttm import write_rttm
from diarisation.runlog import RunLog
from diarisation.systems import pyannote_sys


def audio_jobs(ds, cond, part, audio_root):
    """[(uri, wav path)]: AMI files from the corpus layout, or local files when
    the dataset config gives `audio_path` (e.g. the Supreme Court set)."""
    if "audio_path" in ds:
        uris = (REPO_ROOT / ds["out_dir"] / "lists" / f"{part}.txt").read_text().split()
        return [(u, audio_root / ds["audio_path"].format(uri=u)) for u in uris]
    return [(uri, audio_root / "amicorpus" / uri / "audio" / url.rsplit("/", 1)[1])
            for uri, url in audio_urls(ds, cond, [part])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+")
    ap.add_argument("--dataset", help="override the system config's dataset (e.g. configs/data/scotus_oyez.yaml)")
    ap.add_argument("--partitions", nargs="+")
    ap.add_argument("--limit", type=int, default=None, help="first N meetings only (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.dataset:
        cfg["dataset"] = args.dataset
    ds = load_yaml(cfg["dataset"])
    audio_root = load_paths()["audio_root"]

    import numpy as np
    import torch
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    backend = cfg.get("backend", "pyannote")
    if backend == "pyannote":
        pipe, device = pyannote_sys.load(cfg["model"], cfg["expect_pyannote_major"], cfg["device"], cfg["cpu_threads"])
        revision = pyannote_sys.model_revision(cfg["model"])
        run = lambda wav, out, uri: pyannote_sys.diarise(pipe, wav)  # noqa: E731
        packages = ("pyannote.audio", "torch", "torchaudio", "numpy", "scipy")
    elif backend == "nemo_sortformer":
        from diarisation.systems import sortformer_sys
        model, device, revision = sortformer_sys.load(cfg)
        run = lambda wav, out, uri: sortformer_sys.diarise(model, wav, out / f"{uri}.probs.npz")  # noqa: E731
        packages = ("nemo_toolkit", "torch", "torchaudio", "numpy", "huggingface_hub")
    else:
        print(f"unknown backend {backend!r}", file=sys.stderr)
        return 2
    print(f"loaded {cfg['model']} @ {revision[:10]} on {device}", flush=True)

    for cond in args.conditions or cfg["conditions"]:
        for part in args.partitions or cfg["partitions"]:
            out = REPO_ROOT / cfg["out_dir"] / cond / part
            log = RunLog(f"diarise_{cfg['system']}", {"system": cfg, "condition": cond, "partition": part},
                         seed=cfg["seed"], packages=packages)
            jobs = audio_jobs(ds, cond, part, audio_root)[: args.limit]
            timing_p = out / "timing.csv"
            out.mkdir(parents=True, exist_ok=True)
            rows = list(csv.DictReader(open(timing_p))) if timing_p.exists() else []
            done = {r["uri"] for r in rows}
            for i, (uri, wav) in enumerate(jobs, 1):
                rttm = out / f"{uri}.rttm"
                if rttm.exists() and uri in done and not args.force:
                    continue
                if not wav.exists():
                    print(f"missing audio {wav}; run scripts/download_ami_audio.py", file=sys.stderr)
                    return 1
                import soundfile as sf
                dur = sf.info(str(wav)).duration
                t0 = time.perf_counter()
                segs = run(str(wav), out, uri)
                wall = time.perf_counter() - t0
                segs = pyannote_sys.round_segments(segs, cfg["rttm_decimals"])
                write_rttm(rttm, uri, segs)
                rows = [r for r in rows if r["uri"] != uri] + [{
                    "uri": uri, "audio_s": f"{dur:.3f}", "wall_s": f"{wall:.2f}", "rtf": f"{wall / dur:.4f}",
                    "n_speakers": len({s.speaker for s in segs}), "n_segments": len(segs)}]
                with open(timing_p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
                print(f"[{cond}/{part} {i}/{len(jobs)}] {uri} {dur/60:.1f} min audio, {wall:.0f} s wall, "
                      f"RTF {wall/dur:.3f}, {len({s.speaker for s in segs})} speakers", flush=True)
            log.finish(out, model=cfg["model"], model_revision=revision, device=device,
                       n_files=len(rows), **pyannote_sys.peak_memory(device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
