from qrmfold import QuantumReedMuller


import pytest


@pytest.fixture
def qrm4():
    return QuantumReedMuller(4)