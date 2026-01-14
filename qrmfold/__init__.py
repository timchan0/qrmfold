from qrmfold import logical_index_to_subset_maps
from qrmfold.quantum_reed_muller import QuantumReedMuller


if __name__ == "__main__":
    m = 4
    qrm = QuantumReedMuller(
        m,
        logical_index_to_subset=logical_index_to_subset_maps.alternative(m, start_index=1),
    )
    pairs = [(1, 2)]
    physical_circuit = qrm.q_automorphism_phase_type_product(pairs)