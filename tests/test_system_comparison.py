import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_systems import compare  # noqa: E402
from wer_decomposition import systems_of  # noqa: E402

from diarisation.io.rttm import Segment
from diarisation.systems.sortformer_sys import _parse

BS = {"n_resamples": 200, "ci_level": 0.95, "seed": 1}


def rate(s):
    return s["err"] / s["n"]


def test_compare_identical_systems_gives_zero_and_all_ties():
    cols = {"err": [1.0, 2.0, 3.0], "n": [10.0, 10.0, 10.0]}
    row = compare(cols, cols, ["a", "b", "c"], rate, BS)
    assert float(row["difference"]) == 0.0
    assert row["n_ties"] == 3 and row["n_contender_lower"] == 0 and row["n_undefined"] == 0


def test_compare_pools_before_dividing_and_counts_meetings():
    base = {"err": [2.0, 2.0, 2.0], "n": [10.0, 10.0, 20.0]}
    cont = {"err": [1.0, 3.0, 1.0], "n": [10.0, 10.0, 20.0]}
    row = compare(base, cont, ["a", "b", "c"], rate, BS)
    assert float(row["baseline"]) == pytest.approx(6 / 40)
    assert float(row["contender"]) == pytest.approx(5 / 40)
    assert float(row["difference"]) == pytest.approx(-1 / 40)
    assert row["n_contender_lower"] == 2 and row["n_ties"] == 0


def test_compare_marks_meetings_where_the_statistic_is_undefined():
    base = {"err": [1.0, 0.0], "n": [10.0, 0.0]}
    cont = {"err": [2.0, 0.0], "n": [10.0, 0.0]}
    row = compare(base, cont, ["a", "b"], rate, BS)
    assert row["n_undefined"] == 1 and row["n_contender_lower"] == 0


def test_sortformer_segments_parse_from_strings_and_tuples():
    assert _parse("0.00 1.52 speaker_0") == Segment(0.0, 1.52, "speaker_0")
    assert _parse((3.2, 4.0, 2)) == Segment(3.2, 4.0, "speaker_2")
    with pytest.raises(ValueError):
        _parse("0.00 1.52")


def test_one_pyannote_dir_keeps_the_original_config_form():
    assert systems_of({"pyannote_dir": "x"}) == {"pyannote": "x"}
    assert systems_of({"systems": {"pyannote": "x", "sortformer": "y"}}) == {"pyannote": "x", "sortformer": "y"}
