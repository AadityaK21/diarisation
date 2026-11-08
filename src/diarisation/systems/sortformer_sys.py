"""NVIDIA NeMo streaming Sortformer (end-to-end, overlap-aware, at most 4
speakers), loaded from a pinned Hugging Face revision of its .nemo file.

Runs in its own environment (env/requirements-nemo-cpu.txt). The model's own
`diarize()` is used with its default post-processing (no thresholds tuned on
AMI), and the streaming setting is taken from the config."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..io.rttm import Segment


def load(cfg: Dict[str, Any]) -> Tuple[Any, str, str]:
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    torch.set_num_threads(cfg["cpu_threads"])
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = cfg.get("revision") or HfApi().model_info(cfg["model"]).sha
    path = hf_hub_download(cfg["model"], cfg["nemo_file"], revision=revision)
    model = SortformerEncLabelModel.restore_from(restore_path=path, map_location=device, strict=False)
    model.eval()
    sm = model.sortformer_modules
    for k, v in cfg["streaming"].items():
        if not hasattr(sm, k):
            raise RuntimeError(f"Sortformer has no streaming parameter {k!r}")
        setattr(sm, k, v)
    sm._check_streaming_parameters()
    return model, device, revision


def _parse(seg: Any) -> Segment:
    """One output segment: 'start end speaker_N' or a (start, end, speaker) tuple."""
    parts = seg.split() if isinstance(seg, str) else list(seg)
    if len(parts) != 3:
        raise ValueError(f"unexpected Sortformer segment {seg!r}")
    start, end, spk = float(parts[0]), float(parts[1]), str(parts[2])
    if not spk.startswith("speaker"):
        spk = f"speaker_{spk}"
    return Segment(start, end, spk)


def diarise(model, wav_path: str, probs_path=None) -> List[Segment]:
    """Segments from the model's default post-processing. With `probs_path`,
    the frame-level speaker probabilities (80 ms frames x 4 speakers, before
    any post-processing) are also saved there, so other published
    post-processing settings can be applied later without rerunning the model."""
    import numpy as np
    import torch

    with torch.inference_mode():
        lines, preds = model.diarize(audio=[wav_path], batch_size=1, include_tensor_outputs=True)
    if probs_path is not None:
        p = preds[0].squeeze(0).float().numpy()
        np.savez_compressed(probs_path, probs=p.astype(np.float32),
                            frame_s=np.float32(model._cfg.encoder.subsampling_factor * 0.01))
    return [_parse(s) for s in lines[0]]
