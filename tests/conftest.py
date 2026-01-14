from qrmfold.quantum_reed_muller import QuantumReedMuller

import pytest


@pytest.fixture
def qrms():
    return {m: QuantumReedMuller(m) for m in range(2, 12, 2)}