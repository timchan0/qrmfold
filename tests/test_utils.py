import pytest

from qrmfold.utils import all_bitstrings


class TestAllBitstrings:

    @pytest.mark.parametrize("m", range(1, 6))
    def test_correct_count(self, m):
        bitstrings = all_bitstrings(m)
        assert len(bitstrings) == 2**m