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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+")
    ap.add_argument("--partitions", nargs="+")
    ap.add_argument("--limit", type=int, default=None, help="first N meetings only (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ds = load_yaml(cfg["dataset"])
    audio_root = load_paths()["audio_root"] / "amicorpus"

    import numpy as np
    import torch
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    pipe, device = pyannote_sys.load(cfg["model"], cfg["expect_pyannote_major"], cfg["device"], cfg["cpu_threads"])
    revision = pyannote_sys.model_revision(cfg["model"])
    print(f"loaded {cfg['model']} @ {revision[:10]} on {device}", flush=True)

    for cond in args.conditions or cfg["conditions"]:
        for part in args.partitions or cfg["partitions"]:
            out = REPO_ROOT / cfg["out_dir"] / cond / part
            log = RunLog(f"diarise_{cfg['system']}", {"system": cfg, "condition": cond, "partition": part},
                         seed=cfg["seed"], packages=("pyannote.audio", "torch", "torchaudio", "numpy", "scipy"))
            jobs = audio_urls(ds, cond, [part])[: args.limit]
            timing_p = out / "timing.csv"
            out.mkdir(parents=True, exist_ok=True)
            rows = list(csv.DictReader(open(timing_p))) if timing_p.exists() else []
            done = {r["uri"] for r in rows}
            for i, (uri, url) in enumerate(jobs, 1):
                rttm = out / f"{uri}.rttm"
                if rttm.exists() and uri in done and not args.force:
                    continue
                wav = audio_root / uri / "audio" / url.rsplit("/", 1)[1]
                if not wav.exists():
                    print(f"missing audio {wav}; run scripts/download_ami_audio.py", file=sys.stderr)
                    return 1
                import soundfile as sf
                dur = sf.info(str(wav)).duration
                t0 = time.perf_counter()
                segs = pyannote_sys.diarise(pipe, str(wav))
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
