# qrmfold (Quantum Reed-Muller Fold)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://timchan0.github.io/qrmfold/)

A Python package to generate the full logical Clifford group
from transversal and fold-transversal gates in the quantum Reed-Muller code.

## Installation Instructions
Run `pip install qrmfold`.

## Tutorial
See [`getting_started.ipynb`](https://github.com/timchan0/qrmfold/blob/main/getting_started.ipynb).
For more detail on individual functions/methods,
inspect their docstrings.

## Documentation (Sphinx)
Build the HTML docs from the source docstrings:

- Install docs dependencies: `pip install -e ".[docs]"`
- Build: `cd docs && make html`
- Open: `docs/build/html/index.html`