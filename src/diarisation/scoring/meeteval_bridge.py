"""Small bridge to MeetEval: one SegLST segment per token, and the DI-cpWER
assignment turned back into per-speaker token streams.

DI-cpWER (MeetEval, greedy) assigns every hypothesis segment to the reference
speaker that minimises the total error; with one segment per output word it
is the error left when speaker attribution is free: transcription only. The
greedy search starts from the cpWER assignment of the hypothesis labels, so
it is run from several labellings and the lowest result is kept (each run is
an upper bound on the exact DI-cpWER).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

from .wer import Token


def seglst(streams: Dict[str, List[Token]], uri: str):
    from meeteval.io.seglst import SegLST
    return SegLST([{"session_id": uri, "speaker": spk, "start_time": t.start, "end_time": t.end, "words": t.text}
                   for spk, toks in streams.items() for t in toks])


def di_streams(ref_streams: Dict[str, List[Token]], labellings: Sequence[Tuple[str, Dict[str, List[Token]]]],
               uri: str):
    """-> (streams keyed by reference speaker, MeetEval DICPErrorRate, name of the best start)."""
    from meeteval.wer import greedy_di_cp_word_error_rate
    ref_sl = seglst(ref_streams, uri)
    best = None
    for name, streams in labellings:
        hyp_sl = seglst(streams, uri)
        di = greedy_di_cp_word_error_rate(ref_sl, hyp_sl)
        if best is None or di.errors < best[0].errors:
            best = (di, hyp_sl, name)
    di, hyp_sl, name = best
    _, reassigned = di.apply_assignment(ref_sl, hyp_sl)
    out: Dict[str, List[Token]] = {}
    for s in reassigned:
        out.setdefault(s["speaker"], []).append(Token(s["words"], s["start_time"], s["end_time"]))
    # MeetEval orders each stream by time; tokens of one word share their
    # times and keep their order (stable sort)
    out = {k: sorted(v, key=lambda t: (t.start, t.end)) for k, v in out.items()}
    return out, di, name


def ref_token_ops(ref: List[Token], hyp: List[Token]) -> List[str]:
    """Per reference token: 'ok', 'sub' or 'del' under a Levenshtein alignment."""
    ops = ["ok"] * len(ref)
    for op in Levenshtein.editops([t.text for t in ref], [t.text for t in hyp]):
        if op.tag == "replace":
            ops[op.src_pos] = "sub"
        elif op.tag == "delete":
            ops[op.src_pos] = "del"
    return ops
