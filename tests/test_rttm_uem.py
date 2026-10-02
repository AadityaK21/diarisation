import pytest

from diarisation.io.rttm import Segment, read_rttm, write_rttm
from diarisation.io.uem import read_uem, write_uem


def test_rttm_roundtrip(tmp_path):
    p = tmp_path / "x.rttm"
    write_rttm(p, "m1", [Segment(1.5, 2.25, "B"), Segment(0.0, 1.0, "A")])
    got = read_rttm(p)["m1"]
    assert [(s.start, s.end, s.speaker) for s in got] == [(0.0, 1.0, "A"), (1.5, 2.25, "B")]


def test_rttm_rejects_spaces_and_short_lines(tmp_path):
    with pytest.raises(ValueError):
        write_rttm(tmp_path / "a.rttm", "m", [Segment(0, 1, "John Roberts")])
    bad = tmp_path / "b.rttm"
    bad.write_text("SPEAKER m 1 0.0 1.0\n")
    with pytest.raises(ValueError):
        read_rttm(bad)


def test_uem_roundtrip(tmp_path):
    p = tmp_path / "x.uem"
    write_uem(p, "m1", [(0.0, 2142.709)])
    assert read_uem(p) == {"m1": [(0.0, 2142.709)]}
