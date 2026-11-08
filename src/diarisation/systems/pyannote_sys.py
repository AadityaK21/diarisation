"""pyannote/speaker-diarization-3.1, loaded with a clear failure on gated access."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..io.rttm import Segment

GATED_HELP = """\
Could not load {model}. It is gated on Hugging Face. To fix:
  1. Log in at https://huggingface.co and accept the conditions on BOTH
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  2. Create a read token at https://huggingface.co/settings/tokens
  3. export HF_TOKEN=<token>   (or save it as a network secret for huggingface.co)
Underlying error: {err}"""


def load(model: str, expect_major: int, device: str, threads: int):
    import torch
    import pyannote.audio
    from pyannote.audio import Pipeline

    major = int(pyannote.audio.__version__.split(".")[0])
    if major != expect_major:
        raise RuntimeError(f"pyannote.audio {pyannote.audio.__version__} installed, "
                           f"config expects {expect_major}.x")
    torch.set_num_threads(threads)
    # A token in the environment is used if present; otherwise the request goes
    # out unauthenticated and a network-level credential may be injected.
    token = os.environ.get("HF_TOKEN") or True
    try:
        pipe = Pipeline.from_pretrained(model, use_auth_token=token)
    except Exception as exc:  # noqa: BLE001 - re-raised with instructions
        raise RuntimeError(GATED_HELP.format(model=model, err=exc)) from exc
    if pipe is None:
        raise RuntimeError(GATED_HELP.format(model=model, err="from_pretrained returned None"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(torch.device(device))
    return pipe, device


def model_revision(model: str) -> str:
    """Commit hash of the downloaded pipeline config, from the local HF cache."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model).sha or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def diarise(pipe, wav_path: str) -> List[Segment]:
    import soundfile as sf
    import torch

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    if waveform.shape[0] > 1:  # downmix defensively; AMI mix/SDM files are mono
        waveform = waveform.mean(dim=0, keepdim=True)
    ann = pipe({"waveform": waveform, "sample_rate": sr})
    return [Segment(float(s.start), float(s.end), str(lab))
            for s, _, lab in ann.itertracks(yield_label=True)]


def round_segments(segs: List[Segment], decimals: int) -> List[Segment]:
    """Round to the grid and merge same-speaker pieces that now touch or overlap."""
    by: Dict[str, List[tuple]] = {}
    for s in segs:
        a, b = round(s.start, decimals), round(s.end, decimals)
        if b > a:
            by.setdefault(s.speaker, []).append((a, b))
    out: List[Segment] = []
    for spk, regions in by.items():
        regions.sort()
        cur = list(regions[0])
        for a, b in regions[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                out.append(Segment(cur[0], cur[1], spk))
                cur = [a, b]
        out.append(Segment(cur[0], cur[1], spk))
    return sorted(out, key=lambda s: (s.start, s.speaker))


def peak_memory(device: str) -> Dict[str, Any]:
    if device == "cuda":
        import torch
        return {"cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    import resource
    return {"cpu_max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
