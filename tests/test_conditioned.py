import math

from diarisation.io.rttm import Segment as S
from diarisation.scoring.conditioned import decompose, partition_residual
from diarisation.scoring.der import frame_arrays
from diarisation.stats.bootstrap import of_sums

STEP = 0.001
REF = [S(0, 4, "A"), S(3, 6, "B")]
HYP = [S(0, 5, "x"), S(5, 7, "y")]
UEM = [(0.0, 10.0)]


def c(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


def test_regions_by_hand_collar0():
    # 3-4 overlap: miss 1 (2 ref, 1 hyp) | 4-5 single: confusion 1 | 6-7 nonspeech: FA 1
    p = decompose(frame_arrays(REF, HYP, UEM, 0.0, False, STEP))
    assert c(p["overlap"].scored_speaker_s, 2) and c(p["overlap"].missed_s, 1)
    assert c(p["single"].scored_speaker_s, 5) and c(p["single"].confusion_s, 1)
    assert c(p["nonspeech"].scored_speaker_s, 0) and c(p["nonspeech"].false_alarm_s, 1)
    assert partition_residual(p) < 1e-9


def test_collar_applied_after_region_split():
    # Overlap 3-4 with collars at 3 and 4: only 3.25-3.75 of it is scored.
    p = decompose(frame_arrays(REF, HYP, UEM, 0.25, False, STEP))
    assert c(p["overlap"].scored_speaker_s, 2 * 0.5) and c(p["overlap"].missed_s, 0.5)
    assert partition_residual(p) < 1e-9


def test_excluded_overlap_region_is_empty():
    p = decompose(frame_arrays(REF, HYP, UEM, 0.0, True, STEP))
    assert p["overlap"].scored_speaker_s == 0 and math.isnan(p["overlap"].der)


def test_paired_difference_bootstrap_is_zero_for_identical_columns():
    cols = {"a": [1, 2, 3, 4], "b": [10, 10, 10, 10]}
    e = of_sums(cols, lambda s: s["a"] / s["b"] - s["a"] / s["b"], 500, 0.95, 1)
    assert e.point == 0 and e.lo == 0 and e.hi == 0
