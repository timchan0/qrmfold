import pytest
import numpy as np

from qrmfold import QuantumReedMuller, rref_gf2

sign_to_power = {1: 0, 1j: 1, -1: 2, -1j: 3}


class TestCodespacePreservation:

    @pytest.mark.parametrize("gate_type", ['swap', 'phase'])
    @pytest.mark.parametrize("automorphism_type", ['P', 'Q'])
    @pytest.mark.parametrize("m", range(2, 10, 2))
    def test_1_pair(self, m, automorphism_type, gate_type):
        qrm = QuantumReedMuller(m)
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
    def test_1_to_maximal_pair_count(self, m, automorphism_type, gate_type):
        qrm = QuantumReedMuller(m)
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
    def test_trivial_automorphism_phase_type(self, m):
        qrm = QuantumReedMuller(m)
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