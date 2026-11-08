import math
import sys
from pathlib import Path

from diarisation.scoring.meeteval_bridge import di_streams, ref_token_ops
from diarisation.scoring.wer import Token

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from overlap_words import level_bin, region_of  # noqa: E402

T = lambda w, s: Token(w, s, s + 0.2)  # noqa: E731


def test_ref_token_ops_marks_deletions_and_substitutions():
    ref = [T("a", 0), T("b", 1), T("c", 2), T("d", 3)]
    hyp = [T("a", 0), T("x", 2), T("d", 3)]
    ops = ref_token_ops(ref, hyp)
    assert ops.count("ok") == 2 and ops.count("sub") == 1 and ops.count("del") == 1


def test_di_streams_frees_words_from_wrong_labels():
    ref = {"A": [T("hello", 0), T("world", 1)], "B": [T("yeah", 0.5)]}
    wrong = {"x": [T("hello", 0), T("yeah", 0.5), T("world", 1)]}   # everything on one label
    right = {"x": [T("hello", 0), T("world", 1)], "y": [T("yeah", 0.5)]}
    streams, di, start = di_streams(ref, [("wrong", wrong), ("right", right)], "s")
    assert di.errors == 0 and start in ("wrong", "right")   # equal results keep the first start
    assert [t.text for t in streams["A"]] == ["hello", "world"] and [t.text for t in streams["B"]] == ["yeah"]


def test_region_of_and_level_bin():
    regs = [(0.0, 1.0), (2.0, 3.0)]
    assert region_of(regs, 2.5) == (2.0, 3.0) and region_of(regs, 1.5) is None and region_of(regs, 3.0) is None
    edges = [-10, -5, 0, 5, 10]
    assert level_bin(-12, edges) == "[-inf,-10)" and level_bin(0, edges) == "[0,5)"
    assert level_bin(10, edges) == "[10,inf)" and level_bin(math.nan, edges) == "unknown"
