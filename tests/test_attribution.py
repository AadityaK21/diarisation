from diarisation.data.ami_words import Word as W
from diarisation.io.rttm import Segment as S
from diarisation.scoring.attribution import assign, mappings, score_words

STEP = 0.001
UEM = [(0.0, 20.0)]
# A talks 0-10, B 8-14 (overlap 8-10). The system sees x 0-10 and y 10-14:
# it misses B in the overlap, so B's overlapped words go to x (= A).
REF = [S(0, 10, "A"), S(8, 14, "B")]
HYP = [S(0, 10, "x"), S(10, 14, "y")]
WORDS = [W("A", 1, 2, "a1"), W("A", 8.2, 8.6, "a2"), W("B", 8.5, 9.0, "b1"),
         W("B", 11, 12, "b2"), W("B", 15, 15.5, "b3")]


def test_assign_by_max_overlap_and_unassigned():
    assert assign(WORDS, HYP) == ["x", "x", "x", "y", None]


def test_outcomes_by_region():
    m = mappings(REF, HYP, UEM, STEP)
    assert m["optimal"] == {"x": "A", "y": "B"} == m["majority_single"]
    c = score_words(WORDS, REF, HYP, UEM, m["optimal"])
    assert c["single"]["correct"] == 2 and c["single"]["unassigned"] == 1     # a1, b2 ; b3
    assert c["overlap"]["correct"] == 1 and c["overlap"]["other_active_missed"] == 1  # a2 ; b1 -> A


def test_majority_mapping_is_many_to_one():
    hyp = [S(0, 5, "x"), S(5, 10, "z"), S(10, 14, "y")]   # A split into two clusters
    m = mappings(REF, hyp, UEM, STEP)
    assert m["majority_single"] == {"x": "A", "z": "A", "y": "B"}
    assert len(m["optimal"]) == 2                            # one-to-one leaves a cluster unmapped


def test_rule_error_with_perfect_diarisation():
    # The reference itself as the system: B's overlapped word is covered by
    # both A's and B's turns, and the tie goes to the label sorting first (A).
    c = score_words([W("B", 8.5, 9.0, "b1")], REF, REF, UEM, {"A": "A", "B": "B"})
    assert c["overlap"]["other_active_rule"] == 1
