import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from consistency import sign_test  # noqa: E402


def test_sign_test_exact_values():
    assert sign_test(16, 16) == 2 / 2 ** 16
    assert sign_test(8, 16) == 1.0
    assert abs(sign_test(13, 16) - 0.0213) < 1e-4
