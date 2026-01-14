import itertools
import pytest
import numpy as np
import stim
from typing import Literal

from qrmfold import QuantumReedMuller, rref_gf2, sign_to_power


@pytest.fixture
def qrms():
    return {m: QuantumReedMuller(m) for m in range(2, 12, 2)}


class TestCodespacePreservation:

    @pytest.mark.parametrize("gate_type", ['swap', 'phase'])
    @pytest.mark.parametrize("automorphism_type", ['P', 'Q'])
    @pytest.mark.parametrize("m", range(2, 10, 2))
    def test_1_pair(self, m, automorphism_type, gate_type, qrms):
        qrm = qrms[m]
        stabilizer_generators_bsf_rref = rref_gf2(qrm.stabilizer_generators_bsf())
        for k in range(0, m, 2):
            circuit = getattr(qrm.automorphism(automorphism_type, [(k+1, k+2)]), f"{gate_type}_type_circuit")()
            new_stabilizers_bsf: list[np.ndarray] = []
            for basis_generators in qrm.stabilizer_generators.values():
                for generator in basis_generators:
                    morphed = generator.after(circuit)
                    assert (sign_to_power[morphed.sign] + str(morphed).count('Y')) % 4 == 0, f"Failed for (i, j) = {(k+1, k+2)}"
                    xs, zs =  morphed.to_numpy()
                    new_stabilizers_bsf.append(np.append(xs, zs))
            new_stabilizers_bsf_rref = rref_gf2(np.array(new_stabilizers_bsf))
            assert np.array_equal(stabilizer_generators_bsf_rref, new_stabilizers_bsf_rref)

    @pytest.mark.parametrize("gate_type", ['swap', 'phase'])
    @pytest.mark.parametrize("automorphism_type", ['P', 'Q'])
    @pytest.mark.parametrize("m", range(2, 12, 2))
    def test_1_to_maximal_pair_count(self, m, automorphism_type, gate_type, qrms):
        qrm = qrms[m]
        stabilizer_generators_bsf_rref = rref_gf2(qrm.stabilizer_generators_bsf())
        for pair_count in range(1, m//2 + 1):
            pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
            circuit = getattr(qrm.automorphism(automorphism_type, pairs), f"{gate_type}_type_circuit")()
            new_stabilizers_bsf: list[np.ndarray] = []
            for basis_generators in qrm.stabilizer_generators.values():
                for generator in basis_generators:
                    morphed = generator.after(circuit)
                    assert (sign_to_power[morphed.sign] + str(morphed).count('Y')) % 4 == 0, f"Failed for pairs = {pairs}"
                    xs, zs =  morphed.to_numpy()
                    new_stabilizers_bsf.append(np.append(xs, zs))
            new_stabilizers_bsf_rref = rref_gf2(np.array(new_stabilizers_bsf))
            assert np.array_equal(stabilizer_generators_bsf_rref, new_stabilizers_bsf_rref)


    @pytest.mark.parametrize("m", range(2, 10, 2))
    def test_trivial_automorphism_phase_type(self, m, qrms):
        qrm = qrms[m]
        stabilizer_generators_bsf_rref = rref_gf2(qrm.stabilizer_generators_bsf())
        circuit = qrm.trivial_automorphism().phase_type_circuit()
        new_stabilizers_bsf: list[np.ndarray] = []
        for basis_generators in qrm.stabilizer_generators.values():
            for generator in basis_generators:
                morphed = generator.after(circuit)
                assert (sign_to_power[morphed.sign] + str(morphed).count('Y')) % 4 == 0
                xs, zs =  morphed.to_numpy()
                new_stabilizers_bsf.append(np.append(xs, zs))
        new_stabilizers_bsf_rref = rref_gf2(np.array(new_stabilizers_bsf))
        assert np.array_equal(stabilizer_generators_bsf_rref, new_stabilizers_bsf_rref)


class TestLogicalAction:

    @pytest.mark.parametrize(
            "m, pair_count",
            [(m, pair_count) for m in range(2, 12, 2) for pair_count in range(m//2 + 1)],
    )
    def test_single_unitary(self, m: int, pair_count: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
        physical_circuit = qrm.automorphism('Q', pairs).phase_type_circuit()
        realized_tableau = qrm._get_logical_tableau(physical_circuit)
        target_tableau = qrm.q_automorphism_phase_type_logical_action(pairs).to_tableau()
        assert realized_tableau == target_tableau

    @pytest.mark.parametrize(
            "m, pair_count",
            [(m, pair_count) for m in range(2, 12, 2) for pair_count in range(m//2 + 1)],
    )
    def test_product_of_unitaries(self, m: int, pair_count: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        pairs = [(k+1, k+2) for k in range(0, 2*pair_count, 2)]
        physical_circuit = qrm.q_automorphism_phase_type_product(pairs)
        realized_tableau = qrm._get_logical_tableau(physical_circuit)
        target_tableau = qrm.q_automorphism_phase_type_product_logical_action(pairs).to_tableau()
        assert target_tableau == realized_tableau


class TestLogical2QubitGateRestricted:

    @pytest.mark.parametrize("m", range(2, 8, 2))
    @pytest.mark.parametrize("gate", ['SWAP', 'CZ_XX'])
    def test_matching_tableau(self, m: int, gate: Literal['SWAP', 'CZ_XX'], qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        to_test = qrm._logical_czxx_restricted if gate == 'CZ_XX' else qrm._logical_swap_restricted
        gates = ['CZ', 'Z'] if gate == 'CZ_XX' else ['SWAP']
        for (i, i_tuple), (j, j_tuple) in itertools.combinations(qrm.logical_index_to_subset.items(), 2):
            i_subset = set(i_tuple)
            j_subset = set(j_tuple)
            if len(i_subset.intersection(j_subset)) == qrm.M//2 - 1:
                
                physical_circuit = to_test(i_subset, j_subset)
                realized_tableau = qrm._get_logical_tableau(physical_circuit)
            
                target_circuit = stim.Circuit()
                target_circuit.append('I', qrm.logical_index_to_subset.keys(), ())
                for stim_gate in gates:
                    target_circuit.append(stim_gate, [i, j], ())
                target_tableau = target_circuit.to_tableau()
            
                assert target_tableau == realized_tableau


class TestLogical:

    @pytest.mark.parametrize("m", range(2, 8, 2))
    @pytest.mark.parametrize("gate", ['S', 'H'])
    def test_1_qubit_gate(self, gate: Literal['S', 'H'], m: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        for logical_index in qrm.logical_index_to_subset.keys():
            physical_circuit = qrm.logical(gate, [logical_index])
            realized_tableau = qrm._get_logical_tableau(physical_circuit)
            
            target_circuit = stim.Circuit()
            target_circuit.append('I', qrm.logical_index_to_subset.keys(), ())
            target_circuit.append(gate, [logical_index], ())
            target_tableau = target_circuit.to_tableau()
            
            assert target_tableau == realized_tableau

    @pytest.mark.parametrize("m", range(2, 6, 2))
    @pytest.mark.parametrize("gate", ['SWAP', 'CZ_XX'])
    def test_2_qubit_gate(self, gate: Literal['SWAP', 'CZ_XX'], m: int, qrms: dict[int, QuantumReedMuller]):
        qrm = qrms[m]
        gates = ['SWAP'] if gate == 'SWAP' else ['CZ', 'Z']
        for i, j in itertools.combinations(qrm.logical_index_to_subset.keys(), 2):
            physical_circuit = qrm.logical(gate, [i, j])
            realized_tableau = qrm._get_logical_tableau(physical_circuit)
        
            target_circuit = stim.Circuit()
            target_circuit.append('I', qrm.logical_index_to_subset.keys(), ())
            for stim_gate in gates:
                target_circuit.append(stim_gate, [i, j], ())
            target_tableau = target_circuit.to_tableau()
        
            assert target_tableau == realized_tableau