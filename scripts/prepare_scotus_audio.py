"""Supreme Court set: fetch the Oyez argument audio, check its sha256, convert
to 16 kHz mono WAV with ffmpeg (under audio_root, never in the repository),
and write reference words with pseudo timings from the Oyez transcript.

Oyez gives text per text block (a few sentences) with block start/stop, not
per word; each word gets an equal share of its block. These timings are only
used to order words and to give the word rule something to work with; they
are not word alignments (decisions.md X5)."""

import argparse
import hashlib
import json
import subprocess
import sys

import requests

from diarisation.config import REPO_ROOT, load_paths, load_yaml
from diarisation.data.ami_words import Word, write_tsv
from diarisation.data.scotus_oyez import _get


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/scotus_oyez.yaml")
    cfg = load_yaml(ap.parse_args().config)
    paths = load_paths()
    root = paths["audio_root"]
    if REPO_ROOT in root.resolve().parents or root.resolve() == REPO_ROOT:
        sys.exit(f"audio_root {root} is inside the repository; refusing.")
    uris = (REPO_ROOT / cfg["out_dir"] / "lists" / "all.txt").read_text().split()
    a = cfg["audio"]
    for uri in uris:
        wav = root / cfg["audio_path"].format(uri=uri)
        mp3 = wav.with_suffix(".mp3")
        wav.parent.mkdir(parents=True, exist_ok=True)
        if not mp3.exists():
            r = requests.get(a["url"], timeout=300)
            r.raise_for_status()
            mp3.write_bytes(r.content)
        digest = sha256(mp3)
        if digest != a["sha256"]:
            sys.exit(f"{mp3}: sha256 {digest} != pinned {a['sha256']}")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(mp3), *a["ffmpeg_args"], str(wav)],
                           check=True)
        print(f"{uri}: audio ok ({wav})")

        # reference words from the cached (or re-fetched) Oyez transcript
        term, docket = uri.split("_", 1)
        raw = paths["cache_root"] / "oyez" / f"{uri}.json"
        if not raw.exists():
            detail = _get(f"{cfg['source']['api']}/cases/{term}/{docket}")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(_get(detail["oral_argument_audio"][0]["href"])), encoding="utf-8")
        obj = json.loads(raw.read_text(encoding="utf-8"))
        rows = []
        for section in (obj.get("transcript") or {}).get("sections") or []:
            for turn in section.get("turns") or []:
                spk = (turn.get("speaker") or {}).get("name") or "UNKNOWN"
                spk = spk.replace(" ", cfg["speaker_space_replacement"])
                for block in turn.get("text_blocks") or []:
                    ws = (block.get("text") or "").split()
                    s, e = float(block["start"]), float(block["stop"])
                    if e <= s:  # one Oyez block ("The case is submitted.") has stop 0: use its turn
                        s, e = float(turn["start"]), float(turn["stop"])
                    d = (e - s) / len(ws) if ws else 0.0
                    rows += [(spk, s + i * d, s + (i + 1) * d, w) for i, w in enumerate(ws)]
        out = REPO_ROOT / cfg["out_dir"] / "words" / f"{uri}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        # same writer as the AMI words (csv quoting: Oyez text contains quotes)
        write_tsv(out, [Word(spk, s, e, w) for spk, s, e, w in sorted(rows, key=lambda r: (r[1], r[2], r[0]))])
        print(f"{uri}: {len(rows)} reference words, {len({r[0] for r in rows})} speakers -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
