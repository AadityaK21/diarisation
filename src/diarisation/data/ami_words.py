"""Word-level reference for AMI from the NXT manual annotations.

One TSV per meeting: speaker (global AMI id, as in the RTTMs), start, end,
word. The speaker id comes from corpusResources/meetings.xml, which maps each
meeting's channel letter (A-E) to the participant's global name.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..config import REPO_ROOT

NITE = "{http://nite.sourceforge.net/}"


@dataclass(frozen=True)
class Word:
    speaker: str
    start: float
    end: float
    text: str


def ensure_archive(cfg: Dict, cache_root: Path) -> Path:
    dest = cache_root / "ami_nxt"
    zpath = dest / Path(cfg["source"]["url"]).name
    if not zpath.exists():
        dest.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg["source"]["url"], zpath)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != cfg["source"]["sha256"]:
        raise RuntimeError(f"{zpath}: sha256 {digest} != pinned {cfg['source']['sha256']}")
    root = dest / "nxt"
    if not (root / "words").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    return root


def agent_map(nxt_root: Path) -> Dict[str, Dict[str, str]]:
    """{meeting: {agent letter: global speaker id}}"""
    out: Dict[str, Dict[str, str]] = {}
    for m in ET.parse(nxt_root / "corpusResources" / "meetings.xml").getroot().iter("meeting"):
        out[m.get("observation")] = {s.get("nxt_agent"): s.get("global_name") for s in m.iter("speaker")}
    return out


def read_words(nxt_root: Path, meeting: str, agents: Dict[str, str]) -> List[Word]:
    words: List[Word] = []
    for letter, spk in sorted(agents.items()):
        f = nxt_root / "words" / f"{meeting}.{letter}.words.xml"
        if not f.exists():
            continue
        for el in ET.parse(f).getroot():
            if el.tag != "w" or el.get("punc") == "true":
                continue
            s, e = el.get("starttime"), el.get("endtime")
            if s is None or e is None or not (el.text or "").strip():
                continue
            words.append(Word(spk, float(s), float(e), el.text.strip()))
    words.sort(key=lambda w: (w.start, w.end, w.speaker))
    return words


def write_tsv(path: Path, words: List[Word]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["speaker", "start", "end", "word"])
        for x in words:
            w.writerow([x.speaker, f"{x.start:.3f}", f"{x.end:.3f}", x.text])


def read_tsv(path: Path) -> List[Word]:
    with open(path, encoding="utf-8") as fh:
        return [Word(r["speaker"], float(r["start"]), float(r["end"]), r["word"])
                for r in csv.DictReader(fh, delimiter="\t")]


def prepare(cfg: Dict, ami_cfg: Dict, cache_root: Path) -> Dict:
    nxt = ensure_archive(cfg, cache_root)
    agents = agent_map(nxt)
    out = REPO_ROOT / cfg["out_dir"]
    summary = {}
    for part in cfg["partitions"]:
        uris = (REPO_ROOT / ami_cfg["out_dir"] / "lists" / f"{part}.txt").read_text().split()
        n = 0
        for uri in uris:
            words = read_words(nxt, uri, agents[uri])
            write_tsv(out / part / f"{uri}.tsv", words)
            n += len(words)
        summary[part] = {"meetings": len(uris), "words": n}
    (out / "MANIFEST.json").write_text(json.dumps({"source": cfg["source"], "summary": summary}, indent=2) + "\n")
    return summary
