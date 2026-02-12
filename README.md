# qrmfold

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://timchan0.github.io/qrmfold/)

qrmfold (Quantum Reed-Muller Fold)
is the Python package accompanying the paper
[Construction of the full logical Clifford group for high-rate quantum Reed-Muller codes using only transversal and fold-transversal gates](https://arxiv.org/abs/2602.09788).
Its main function is to output
addressable S, Hadamard, controlled-Z, and swap gates
for quantum Reed-Muller codes.
The output format is a [`stim.Circuit`](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference#stim.Circuit)
which can then easily be [converted](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference#stim.Circuit.to_qasm) to QASM.

## Installation Instructions
qrmfold is available as a [PyPI package](https://pypi.org/project/qrmfold/)
so it can be installed by running `pip install qrmfold`.

## Tutorial
See the
[`getting_started.ipynb`](https://github.com/timchan0/qrmfold/blob/main/getting_started.ipynb)
notebook.
For more detail on individual functions/methods,
see below.

## Documentation
If you just want to read the documentation,
click the blue **Documentation** badge at the top of this README.

Optionally, you can build the HTML docs locally:

- Install docs dependencies: `pip install -e ".[docs]"`
- Build: `cd docs && make html`
- Open: `docs/build/html/index.html`

## How to cite qrmfold
Please cite the accompanying paper:
```latex
@misc{tansuwannont2026constructionlogicalcliffordgroup,
      title={Construction of the full logical Clifford group for high-rate quantum Reed-Muller codes using only transversal and fold-transversal gates}, 
      author={Theerapat Tansuwannont and Tim Chan and Ryuji Takagi},
      year={2026},
      eprint={2602.09788},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2602.09788}, 
}
```

## Tests

This package uses the `pytest` framework.
The tests are divided into unit tests and integration tests.
The integration tests verify that:
1. the fold-transversal gates preserve the stabilizer group (Theorems 3 and 4),
2. the fold-transversal gates induce the logical action predicted by Theorems 5 and 6,
3. the addressable gates built from Corollary 1 and Theorems 7 and 8 induce the desired logical action.

Note there is one test, `test_quantum_reed_muller_integration.TestAddressableLogicalAction.test_2_qubit_gate` for the parameter $m =6$,
that takes considerably longer than the others
i.e. about 300 s on a laptop.

## Release (maintainers)
This repo includes a GitHub Actions workflow that publishes to PyPI when a version tag like `v0.7.0` is pushed.