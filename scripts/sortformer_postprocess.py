"""Apply a published post-processing setting to saved Sortformer frame
probabilities (written by diarise.py with configs/systems/sortformer_v21.yaml)
and write one RTTM per meeting. Runs in venv-nemo and uses NeMo's own
post-processing function, exactly as the model's diarize() does.

Check before anything is written: NeMo's default parameters applied to the
saved probabilities must reproduce the RTTMs of the source run exactly, else
the run aborts.

    python scripts/sortformer_postprocess.py --config configs/systems/sortformer_v21_dihard3pp.yaml
"""

import argparse
import csv
import sys

import numpy as np

from diarisation.config import REPO_ROOT, load_yaml
from diarisation.io.rttm import Segment, read_rttm, write_rttm
from diarisation.runlog import RunLog
from diarisation.systems.pyannote_sys import round_segments


def segments(probs, params, frame_count):
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, ts_vad_post_processing
    from omegaconf import OmegaConf

    # the same object NeMo builds from a post-processing YAML (load_postprocessing_from_yaml)
    pp = OmegaConf.structured(PostProcessingParams())
    for k, v in params.items():
        if not hasattr(pp, k):
            raise KeyError(f"unknown post-processing parameter {k!r}")
        setattr(pp, k, v)
    out = []
    for spk in range(probs.shape[1]):
        ts = ts_vad_post_processing(torch.from_numpy(probs[:, spk].copy()), cfg_vad_params=pp,
                                    unit_10ms_frame_count=frame_count, bypass_postprocessing=False)
        # NeMo rounds to 10 ms before writing its segments (sortformer_diar_models.py)
        out += [Segment(round(a, 2), round(b, 2), f"speaker_{spk}") for a, b in ts.tolist()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    src_cfg = load_yaml(cfg["source_system"])
    log = RunLog(cfg["system"], {"system": cfg, "source": src_cfg}, seed=None,
                 packages=("nemo_toolkit", "torch", "numpy"))
    frame_count = cfg["unit_10ms_frame_count"]
    counts = {}
    for cond in cfg["conditions"]:
        for part in cfg["partitions"]:
            src = REPO_ROOT / src_cfg["out_dir"] / cond / part
            out = REPO_ROOT / cfg["out_dir"] / cond / part
            uris = sorted(p.name[: -len(".probs.npz")] for p in src.glob("*.probs.npz"))
            if not uris:
                print(f"ABORT: no probability files in {src}", file=sys.stderr)
                return 2
            rows = []
            for uri in uris:
                z = np.load(src / f"{uri}.probs.npz")
                probs = z["probs"].astype(np.float32)
                # 1. the default setting must give back the source RTTM exactly
                again = round_segments(segments(probs, {}, frame_count), src_cfg["rttm_decimals"])
                orig = read_rttm(src / f"{uri}.rttm").get(uri, [])
                fmt = lambda ss: sorted(f"{x.start:.3f} {x.end - x.start:.3f} {x.speaker}" for x in ss)  # noqa: E731
                if fmt(again) != fmt(orig):
                    print(f"ABORT {cond}/{uri}: default post-processing does not reproduce the source RTTM "
                          f"({len(again)} vs {len(orig)} segments)", file=sys.stderr)
                    return 2
                # 2. the published setting
                segs = round_segments(segments(probs, cfg["postprocessing"], frame_count), src_cfg["rttm_decimals"])
                out.mkdir(parents=True, exist_ok=True)
                write_rttm(out / f"{uri}.rttm", uri, segs)
                speech = lambda ss: sum(s.duration for s in ss)  # noqa: E731
                rows.append({"uri": uri, "n_speakers": len({s.speaker for s in segs}), "n_segments": len(segs),
                             "speaker_s": f"{speech(segs):.3f}", "speaker_s_default": f"{speech(orig):.3f}"})
            with open(out / "segments.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            log.finish(out, condition=cond, partition=part, n_files=len(rows))
            counts[f"{cond}/{part}"] = len(rows)
    print(f"default setting reproduces every source RTTM; wrote {counts} -> {cfg['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
