from qrmfold import _all_bitstrings


def test_correct_count():
    for m in range(1, 6):
        bitstrings = _all_bitstrings(m)
        assert len(bitstrings) == 2**m