from diarisation.stats.bootstrap import mean, ratio_of_sums


def test_ratio_of_sums_point_is_pooled_not_mean_of_ratios():
    e = ratio_of_sums([1, 9], [10, 10], 200, 0.95, 0)
    assert e.point == 0.5
    e = ratio_of_sums([1, 1], [10, 90], 200, 0.95, 0)
    assert abs(e.point - 0.02) < 1e-12


def test_same_seed_same_interval_and_contains_point():
    x = [0.1, 0.2, 0.15, 0.3, 0.05, 0.22]
    a, b = mean(x, 2000, 0.95, 7), mean(x, 2000, 0.95, 7)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.lo <= a.point <= a.hi


def test_single_file_has_no_interval():
    e = ratio_of_sums([1], [10], 100, 0.95, 0)
    assert e.n_files == 1 and e.lo != e.lo  # NaN: one file cannot be resampled
