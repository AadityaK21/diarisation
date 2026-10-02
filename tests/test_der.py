import math
import shutil

import pytest

from diarisation.io.rttm import Segment as S
from diarisation.scoring.der import SelfOverlapError, score

STEP = 0.001
UEM = [(0.0, 10.0)]


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


def test_perfect_up_to_relabel():
    ref = [S(0, 4, "A"), S(3, 6, "B")]
    hyp = [S(0, 4, "x"), S(3, 6, "y")]
    r = score(ref, hyp, UEM, 0.0, False, STEP)
    assert close(r.scored_speaker_s, 7) and r.der == 0


def test_components_by_hand():
    # ref A 0-4, B 3-6 ; hyp x 0-5 (A), y 5-7
    # 0-3: ref{A} hyp{x}          correct
    # 3-4: ref{A,B} hyp{x}        miss 1
    # 4-5: ref{B} hyp{x}          confusion 1 (x->A)
    # 5-6: ref{B} hyp{y}          y maps to B (1 s overlap) -> correct
    # 6-7: ref{} hyp{y}           false alarm 1
    ref = [S(0, 4, "A"), S(3, 6, "B")]
    hyp = [S(0, 5, "x"), S(5, 7, "y")]
    r = score(ref, hyp, UEM, 0.0, False, STEP)
    assert close(r.scored_speaker_s, 7)
    assert close(r.missed_s, 1) and close(r.confusion_s, 1) and close(r.false_alarm_s, 1)


def test_exclude_overlap_and_collar():
    ref = [S(0, 4, "A"), S(3, 6, "B")]
    hyp = [S(0, 5, "x"), S(5, 7, "y")]
    r = score(ref, hyp, UEM, 0.0, True, STEP)      # drops 3-4
    assert close(r.scored_speaker_s, 5) and close(r.missed_s, 0)
    r = score(ref, hyp, UEM, 0.25, False, STEP)    # no-score around 0, 3, 4, 6
    # scored: 0.25-2.75 A ok, 3.25-3.75 overlap miss .5, 4.25-5 conf .75,
    # 5-5.75 ok, 6.25-7 FA .75
    assert close(r.missed_s, 0.5) and close(r.confusion_s, 0.75) and close(r.false_alarm_s, 0.75)


def test_self_overlap_rejected():
    with pytest.raises(SelfOverlapError):
        score([S(0, 2, "A"), S(1, 3, "A")], [], UEM, 0.0, False, STEP)


def test_off_grid_counted():
    r = score([S(0.0, 1.0004, "A")], [S(0, 1, "x")], UEM, 0.0, False, STEP)
    assert r.off_grid == 1


@pytest.mark.skipif(shutil.which("perl") is None, reason="perl not installed")
@pytest.mark.parametrize("collar,excl", [(0.0, False), (0.25, False), (0.0, True), (0.25, True)])
def test_matches_mdeval_on_hand_case(tmp_path, collar, excl):
    from diarisation.io.rttm import write_rttm
    from diarisation.io.uem import write_uem
    from diarisation.scoring import mdeval

    ref = [S(0, 4, "A"), S(3, 6, "B"), S(6.5, 8.2, "A"), S(7.9, 9.1, "C")]
    hyp = [S(0, 5, "x"), S(5, 7, "y"), S(7.1, 9.6, "z"), S(8.0, 8.4, "x")]
    write_rttm(tmp_path / "r.rttm", "f", ref)
    write_rttm(tmp_path / "h.rttm", "f", hyp)
    write_uem(tmp_path / "u.uem", "f", UEM)
    ours = score(ref, hyp, UEM, collar, excl, STEP)
    ref_impl = mdeval.run(tmp_path / "r.rttm", tmp_path / "h.rttm", tmp_path / "u.uem", collar, excl)
    for k in ("scored_speaker_s", "missed_s", "false_alarm_s", "confusion_s"):
        assert abs(getattr(ours, k) - getattr(ref_impl, k)) < 1e-6, k


def test_off_grid_uem_end_is_exact_and_not_flagged():
    # UEM ends mid-frame at 5.0004: the last 0.4 ms of A must count exactly.
    ref = [S(0, 6, "A")]
    r = score(ref, [], [(0.0, 5.0004)], 0.0, False, STEP)
    assert abs(r.scored_speaker_s - 5.0004) < 1e-9 and abs(r.missed_s - 5.0004) < 1e-9
    assert r.off_grid == 0
    r = score(ref, [], [(0.0004, 0.0007)], 0.0, False, STEP)   # inside one frame
    assert abs(r.scored_speaker_s - 0.0003) < 1e-12
