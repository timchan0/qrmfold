import itertools
from typing import Literal

import pytest
import numpy as np
from numpy import typing as npt
import stim

from qrmfold.utils import sign_to_power
from qrmfold.quantum_reed_muller import QuantumReedMuller
from qrmfold.utils import rref_gf2


class TestCodespacePreservation:
    """Test the stabilizer group is preserved."""

    def _helper(
            self,
            qrm: QuantumReedMuller,
            circuit: stim.Circuit,
    ):
        new_stabilizers_bsf: list[npt.NDArray[np.bool_]] = []
        for basis_generators in qrm.stabilizer_generators.values():
            for generator in basis_generators:
                morphed = generator.after(circuit)
                assert (sign_to_power[morphed.sign] + str(morphed).count('Y')) % 4 == 0
                xs, zs =  morphed.to_numpy()
                new_stabilizers_bsf.append(np.append(xs, zs))
        new_stabilizer_generators_rref = rref_gf2(new_stabilizers_bsf)
        assert np.array_equal(new_stabilizer_generators_rref, qrm.stabilizer_generators_rref)

    @pytest.mark.parametrize("gate_type", ['swap', 'phase'])
    @pytest.mark.parametrize("automorphism_type", ['P', 'Q'])
    @pytest.mark.parametrize("m", range(2, 12, 2))
    def test_1_pair(
        self,
        m: int,
        automorphism_type: Literal['P', 'Q'],
        gate_type: Literal['swap', 'phase'],
        qrms: dict[int, QuantumReedMuller],
    ):
        """Test U_S(P(i, j)), U_P(P(i, j), U_S(Q(i, j)), U_P(Q(i, j))
        preserve the codespace.
        """
        qrm = qrms[m]
        for k in range(0, m, 2):
            circuit = qrm.automorphism([(k+1, k+2)], automorphism_type, gate_type)
            self._helper(qrm, circuit)

    @pytest.mark.parametrize("gate_type", ['swap', 'phase'])
    @pytest.mark.parametrize("automorphism_type", ['P', 'Q'])
    @pytest.mark.parametrize("m", range(2, 12, 2))
    def test_1_to_maximal_pair_count(
        self,
        m: int,
        automorphism_type: Literal['P', 'Q'],
        gate_type: Literal['swap', 'phase'],
        qrms: dict[int, QuantumReedMuller],
    ):
        """Test U_S(P(K)), U_P(P(K), U_S(Q(K)), U_P(Q(K))
        preserve the codespace for |K| in 1..m/2.
        """
        qrm = qrms[m]
        for pair_count in range(1, m//2 + 1):
            pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
            circuit = qrm.automorphism(pairs, automorphism_type, gate_type)
            self._helper(qrm, circuit)

    @pytest.mark.parametrize("m", range(2, 12, 2))
    def test_trivial_automorphism_phase_type(self, m: int, qrms: dict[int, QuantumReedMuller]):
        """Test U_P(e) preserves the codespace.
        U_S(e) is identity so need not be tested.
        """
        qrm = qrms[m]
        circuit = qrm.automorphism(gate_type='phase')
        self._helper(qrm, circuit)

    @pytest.mark.parametrize("reduce_depth", (False, True))
    @pytest.mark.parametrize("name", ('ZZCZ', 'SWAP'))
    @pytest.mark.parametrize("m", range(2, 6, 2))
    def test_addressable_gate(
        self,
        m: int,
        name: Literal['ZZCZ', 'SWAP'],
        reduce_depth: bool,
        qrms: dict[int, QuantumReedMuller],
    ):
        qrm = qrms[m]
        for j, k in itertools.combinations(qrm.logical_qubit_ordering.keys(), 2):
            circuit = qrm.gate(name, [j, k], reduce_depth=reduce_depth)
            self._helper(qrm, circuit)


class TestLogicalAction:
    """Test for the correct logical action."""

    @pytest.mark.parametrize(
            "m, pair_count",
            [(m, pair_count) for m in range(2, 12, 2) for pair_count in range(m//2 + 1)],
    )
    def test_single_unitary(self, m: int, pair_count: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
        physical_circuit = qrm.automorphism(pairs, 'Q', 'phase')
        realized_tableau = qrm._get_logical_tableau(physical_circuit)
        target_tableau = qrm.q_phase_logical_action(pairs).to_tableau()
        assert realized_tableau == target_tableau

    @pytest.mark.parametrize(
            "m, pair_count",
            [(m, pair_count) for m in range(2, 12, 2) for pair_count in range(m//2 + 1)],
    )
    def test_product_of_unitaries(self, m: int, pair_count: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
        physical_circuit = qrm.automorphism_product(pairs, automorphism_type='Q', gate_type='phase')
        realized_tableau = qrm._get_logical_tableau(physical_circuit)
        target_tableau = qrm.q_phase_product_logical_action(pairs).to_tableau()
        assert target_tableau == realized_tableau


class TestAddressableLogicalAction:

    @pytest.mark.parametrize("reduce_depth", (False, True))
    @pytest.mark.parametrize("m", range(2, 8, 2))
    @pytest.mark.parametrize("name", ['S', 'H'])
    def test_1_qubit_gate(
        self,
        qrms: dict[int, QuantumReedMuller],
        m: int,
        name: Literal['S', 'H'],
        reduce_depth: bool,
    ):
        qrm = qrms[m]
        for logical_index in qrm.logical_qubit_ordering.keys():
            physical_circuit = qrm.gate(name, [logical_index], reduce_depth=reduce_depth)
            realized_tableau = qrm._get_logical_tableau(physical_circuit)
            
            target_circuit = stim.Circuit()
            target_circuit.append('I', qrm.logical_qubit_ordering.keys(), ())
            target_circuit.append(name, [logical_index], ())
            target_tableau = target_circuit.to_tableau()
            
            assert target_tableau == realized_tableau

    @pytest.mark.parametrize("m", range(2, 8, 2))
    @pytest.mark.parametrize("name", ['SWAP', 'ZZCZ'])
    def test_restricted_2_qubit_gate(
        self,
        qrms: dict[int, QuantumReedMuller],
        m: int,
        name: Literal['SWAP', 'ZZCZ'],
    ):
        qrm = qrms[m]
        to_test = qrm._zzcz_restricted if name == 'ZZCZ' else qrm._swap_restricted
        gates = ['CZ', 'Z'] if name == 'ZZCZ' else ['SWAP']
        for (i, i_tuple), (j, j_tuple) in itertools.combinations(qrm.logical_qubit_ordering.items(), 2):
            i_subset = set(i_tuple)
            j_subset = set(j_tuple)
            if len(i_subset.intersection(j_subset)) == qrm.BIT_COUNT//2 - 1:
                
                physical_circuit = to_test(i_subset, j_subset)
                realized_tableau = qrm._get_logical_tableau(physical_circuit)
            
                target_circuit = stim.Circuit()
                target_circuit.append('I', qrm.logical_qubit_ordering.keys(), ())
                for stim_gate in gates:
                    target_circuit.append(stim_gate, [i, j], ())
                target_tableau = target_circuit.to_tableau()
            
                assert target_tableau == realized_tableau

    @pytest.mark.parametrize("reduce_depth", (False, True))
    @pytest.mark.parametrize("m", range(2, 8, 2))
    @pytest.mark.parametrize("name", ['SWAP', 'ZZCZ'])
    def test_2_qubit_gate(
        self,
        qrms: dict[int, QuantumReedMuller],
        m: int,
        name: Literal['SWAP', 'ZZCZ'],
        reduce_depth: bool,
    ):
        """Test the logical action of either a SWAP or ZZCZ gate
        on any pair of logical qubits.
        
        Note: there are 4 tests for m = 6 which in total take about 300 s on a laptop.
        """
        qrm = qrms[m]
        gates = ['SWAP'] if name == 'SWAP' else ['CZ', 'Z']
        for i, j in itertools.combinations(qrm.logical_qubit_ordering.keys(), 2):
            physical_circuit = qrm.gate(name, [i, j], reduce_depth=reduce_depth)
            realized_tableau = qrm._get_logical_tableau(physical_circuit)
        
            target_circuit = stim.Circuit()
            target_circuit.append('I', qrm.logical_qubit_ordering.keys(), ())
            for stim_gate in gates:
                target_circuit.append(stim_gate, [i, j], ())
            target_tableau = target_circuit.to_tableau()
        
            assert target_tableau == realized_tableau