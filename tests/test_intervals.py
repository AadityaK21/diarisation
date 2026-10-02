import math

from diarisation.intervals import count_profile, count_timeline, merge_regions, outside_uem
from diarisation.io.rttm import Segment as S


def approx(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


def test_merge_regions_touching_and_nested():
    assert merge_regions([(3, 4), (0, 1), (1, 2), (0.5, 0.7)]) == [(0, 2), (3, 4)]


def test_two_speakers_partial_overlap():
    # A 0-4, B 3-6, UEM 0-10: silence 4 s, single 5 s, overlap 1 s.
    p = count_profile([S(0, 4, "A"), S(3, 6, "B")], [(0, 10)])
    assert approx(p[0], 4) and approx(p[1], 5) and approx(p[2], 1)


def test_self_overlap_is_not_overlap():
    p = count_profile([S(0, 4, "A"), S(2, 5, "A")], [(0, 5)])
    assert approx(p[1], 5) and 2 not in p


def test_back_to_back_turns_have_no_overlap():
    # The Oyez case: turns tile the timeline with zero gaps.
    p = count_profile([S(0, 9, "A"), S(9, 131.36, "B"), S(131.36, 162.56, "A")], [(0, 162.56)])
    assert set(p) == {1} and approx(p[1], 162.56)


def test_three_way_overlap_and_uem_clipping():
    segs = [S(0, 6, "A"), S(2, 6, "B"), S(4, 8, "C")]
    p = count_profile(segs, [(1, 7)])
    # 1-2: A | 2-4: A,B | 4-6: A,B,C | 6-7: C
    assert approx(p[1], 2) and approx(p[2], 2) and approx(p[3], 2)
    assert approx(sum(p.values()), 6)
    assert approx(outside_uem(segs, [(1, 7)]), 1 + 1)  # A 0-1, C 7-8


def test_timeline_covers_uem_exactly_with_split_uem():
    segs = [S(0, 10, "A")]
    tl = count_timeline(segs, [(0, 2), (5, 6)])
    assert tl == [(0, 2, 1), (5, 6, 1)]
