"""US Supreme Court reference from Oyez per-turn timings.

Turn parsing follows lausanne/src/eval/fetch_oyez.py::parse_transcript so the
reference is the one the earlier 7.92% DER was scored against. Only the
transcript JSON is fetched; no audio.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from ..config import REPO_ROOT
from ..io.rttm import Segment, write_rttm
from ..io.uem import write_uem

HEADERS = {"User-Agent": "diarisation-research/0.1 (academic)", "Accept": "application/json"}


def _get(url: str, retries: int = 4) -> Any:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def parse_turns(audio_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = []
    for section in (audio_obj.get("transcript") or {}).get("sections") or []:
        for turn in section.get("turns") or []:
            spk = turn.get("speaker") or {}
            name = spk.get("name") or spk.get("last_name") or spk.get("identifier") or "UNKNOWN"
            starts, stops, has_text = [], [], False
            for b in turn.get("text_blocks") or []:
                if not (b.get("text") or "").strip():
                    continue
                has_text = True
                if b.get("start") is not None:
                    starts.append(float(b["start"]))
                if b.get("stop") is not None:
                    stops.append(float(b["stop"]))
            if not has_text:
                continue
            start = min(starts) if starts else turn.get("start")
            stop = max(stops) if stops else turn.get("stop")
            if start is None or stop is None:
                continue
            turns.append({"start": float(start), "stop": float(stop), "speaker": str(name).strip()})
    turns.sort(key=lambda t: t["start"])
    return turns


def prepare(cfg: Dict[str, Any], cache_root: Path) -> Dict[str, Any]:
    out = REPO_ROOT / cfg["out_dir"]
    cache = cache_root / "oyez"
    cache.mkdir(parents=True, exist_ok=True)
    repl = cfg["speaker_space_replacement"]
    uris = []
    summary = {}
    for case in cfg["source"]["cases"]:
        term, docket = case.split("/", 1)
        detail = _get(f"{cfg['source']['api']}/cases/{term}/{docket}")
        args = detail.get("oral_argument_audio") or []
        if not args:
            raise RuntimeError(f"{case}: no oral argument audio listed on Oyez")
        # The first argument session, as lausanne used.
        raw_path = cache / f"{term}_{docket}.json"
        audio_obj = _get(args[0]["href"])
        raw_path.write_text(json.dumps(audio_obj), encoding="utf-8")
        turns = parse_turns(audio_obj)
        uri = f"{term}_{docket.replace('/', '-')}"
        segs = [Segment(t["start"], t["stop"], t["speaker"].replace(" ", repl)) for t in turns]
        write_rttm(out / "rttm" / f"{uri}.rttm", uri, segs)
        # No audio here, so the UEM is the span of the reference turns.
        write_uem(out / "uem" / f"{uri}.uem", uri, [(0.0, max(s.end for s in segs))])
        uris.append(uri)
        summary[uri] = {"case": case, "case_name": detail.get("name"), "turns": len(segs),
                        "speakers": len({s.speaker for s in segs})}
    (out / "lists").mkdir(parents=True, exist_ok=True)
    (out / "lists" / "all.txt").write_text("\n".join(uris) + "\n", encoding="utf-8")
    (out / "MANIFEST.json").write_text(
        json.dumps({"source": cfg["source"], "files": summary,
                    "note": "Raw Oyez turn timings, not hand-corrected."}, indent=2) + "\n",
        encoding="utf-8")
    return summary
