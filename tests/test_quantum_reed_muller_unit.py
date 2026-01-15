import pytest

from qrmfold.quantum_reed_muller import _get_intermediate_subsets
from qrmfold.quantum_reed_muller import QuantumReedMuller, _p_automorphism, _q_automorphism
from qrmfold.utils import sliding_window


def test_p_automorphism():
    swaps = _p_automorphism(4, 1, 2)
    expected_swaps = [('0001', '0010'), ('0101', '0110'), ('1001', '1010'), ('1101', '1110')]
    assert set(swaps) == set(expected_swaps)
    swaps = _p_automorphism(4, 3, 4)
    expected_swaps = [('0100', '1000'), ('0101', '1001'), ('0110', '1010'), ('0111', '1011')]
    assert set(swaps) == set(expected_swaps)

def test_q_automorphism():
    swaps = _q_automorphism(4, 1, 2)
    expected_swaps = [('0010', '0011'), ('0110', '0111'), ('1010', '1011'), ('1110', '1111')]
    assert set(swaps) == set(expected_swaps)
    swaps = _q_automorphism(4, 3, 4)
    expected_swaps = [('1000', '1100'), ('1001', '1101'), ('1010', '1110'), ('1011', '1111')]
    assert set(swaps) == set(expected_swaps)


class TestQuantumReedMuller:

    def test_swap_type(self, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[4]
        swap_gates = qrm.automorphism('P', [(1, 2)])._swap_type()
        assert swap_gates == {(9, 10), (13, 14), (1, 2), (5, 6)}
        swap_gates = qrm.automorphism('P', [(3, 4)])._swap_type()
        assert swap_gates == {(5, 9), (7, 11), (6, 10), (4, 8)}
        swap_gates = qrm.automorphism('Q', [(1, 2)])._swap_type()
        assert swap_gates == {(2, 3), (6, 7), (10, 11), (14, 15)}
        swap_gates = qrm.automorphism('Q', [(3, 4)])._swap_type()
        assert swap_gates == {(11, 15), (9, 13), (10, 14), (8, 12)}

    def test_phase_type(self, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[4]
        cz_gates, s_gates = qrm.automorphism('P', [(1, 2)])._phase_type()
        assert cz_gates == {(9, 10), (13, 14), (1, 2), (5, 6)}
        assert s_gates == {0, 3, 12, 15}.union({8, 11, 4, 7})
        cz_gates, s_gates = qrm.automorphism('P', [(3, 4)])._phase_type()
        assert cz_gates == {(5, 9), (7, 11), (6, 10), (4, 8)}
        assert s_gates == {0, 3, 12, 15}.union({1, 2, 13, 14})
        cz_gates, s_gates = qrm.automorphism('Q', [(1, 2)])._phase_type()
        assert cz_gates == {(2, 3), (6, 7), (10, 11), (14, 15)}
        assert s_gates == {0, 9, 12, 5}.union({8, 1, 4, 13})
        cz_gates, s_gates = qrm.automorphism('Q', [(3, 4)])._phase_type()
        assert cz_gates == {(11, 15), (9, 13), (10, 14), (8, 12)}
        assert s_gates == {0, 3, 5, 6}.union({1, 2, 4, 7})


class TestGetIntermediateSubsets:
    
    def test_empty_list(self):
        """Empty list occurs when subsets differ by only one basis vector."""
        b_subset = {1, 2}
        b_prime_subset = {1, 3}
        assert _get_intermediate_subsets(b_subset, b_prime_subset) == []
    
    @pytest.mark.parametrize(
            "b_subset, b_prime_subset",
            [
                ({1, 2}, {3, 4}),
                ({1, 2, 3}, {2, 3, 4}),
                ({1, 2, 3}, {4, 5, 6}),
            ],
    )
    def test_intermediate_subsets(
            self,
            b_subset: set[int],
            b_prime_subset: set[int],
    ):
        subsets = _get_intermediate_subsets(b_subset, b_prime_subset)
        for a, b in sliding_window([b_subset] + subsets + [b_prime_subset], 2):
            assert len(a.difference(b)) == 1
            assert len(b.difference(a)) == 1