from diarisation.io.rttm import Segment as S
from diarisation.scoring.wer import RegionLookup, Token, cp_assignment, merge_by_mapping, score_assignment, tokens

T = lambda w, s: Token(w, s, s + 0.2)  # noqa: E731


def test_normalised_tokens_drop_fillers_and_expand():
    assert [t.text for t in tokens([("Uh,", 0, 1), ("don't", 1, 2), ("Know.", 2, 3)])] == ["do", "not", "know"]


def test_cp_assignment_and_region_counts():
    ref = [S(0, 10, "A"), S(5, 10, "B")]          # overlap 5-10
    region = RegionLookup(ref, [(0.0, 20.0)])
    R = {"A": [T("a", 1), T("b", 2), T("c", 6)], "B": [T("x", 6), T("y", 7)]}
    H = {"s1": [T("a", 1), T("b", 2), T("c", 6), T("x", 6.1)], "s2": [T("y", 7)]}
    pairs = cp_assignment(R, H)
    assert pairs == {"A": "s1", "B": "s2"}
    acc = score_assignment(R, H, pairs, region)
    # A: one insertion ("x" at 6.1, overlap); B: one deletion ("x" at 6, overlap)
    assert acc["single"] == {"n_ref": 2, "sub": 0, "del": 0, "ins": 0}
    assert acc["overlap"] == {"n_ref": 3, "sub": 0, "del": 1, "ins": 1}


def test_merge_by_mapping_many_to_one():
    H = {"s1": [T("a", 1)], "s2": [T("b", 0.5)], "s3": [T("z", 3)]}
    m = merge_by_mapping(H, {"s1": "A", "s2": "A"})
    assert [t.text for t in m["ref:A"]] == ["b", "a"] and "unmapped:s3" in m


def test_ami_spelled_acronyms_are_joined():
    from diarisation.scoring.wer import join_spelled
    assert join_spelled("L_C_D_") == "LCD"
    assert join_spelled("T_V_s") == "TVs"
    assert join_spelled("A_") == "A"
    assert join_spelled("don't") == "don't" and join_spelled("snake_case_") == "snake_case_"
    assert [t.text for t in tokens([("L_C_D_", 0.0, 1.0)])] == [t.text for t in tokens([("LCD", 0.0, 1.0)])] == ["lcd"]
