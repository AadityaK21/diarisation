import pytest

from diarisation.io.rttm import Segment as S
from diarisation.stats.count_confusion import capped, count_confusion


def test_count_confusion_partitions_the_uem():
    ref = [S(0, 4, "a"), S(2, 6, "b")]              # 0-2: 1, 2-4: 2, 4-6: 1, 6-8: 0
    hyp = [S(0, 3, "x"), S(5, 8, "x"), S(3, 4, "y")]  # 0-3: 1, 3-4: 1, 4-5: 0, 5-8: 1
    conf = count_confusion(ref, hyp, [(0, 8)])
    assert sum(conf.values()) == pytest.approx(8)
    assert conf[(2, 1)] == pytest.approx(2)        # overlap 2-4, system shows one speaker
    assert conf[(1, 0)] == pytest.approx(1)        # 4-5 missed
    assert conf[(0, 1)] == pytest.approx(2)        # 6-8 false alarm
    assert conf[(1, 1)] == pytest.approx(3)


def test_count_confusion_detects_overlap_and_caps():
    ref = [S(0, 2, "a"), S(0, 2, "b"), S(0, 2, "c")]
    hyp = [S(0, 1, "x"), S(0, 1, "y")]
    conf = capped(count_confusion(ref, hyp, [(0, 2)]), cap=2)
    assert conf[(2, 2)] == pytest.approx(1) and conf[(2, 0)] == pytest.approx(1)
