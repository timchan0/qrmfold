# qrmfold (Quantum Reed-Muller Fold)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://timchan0.github.io/qrmfold/)

A Python package to construct addressable gates
for self-dual quantum Reed-Muller codes
as a sequence of transversal and fold-transversal gates.

## Installation Instructions
Run `pip install qrmfold`.

## Tutorial
See the
[`getting_started.ipynb`](https://github.com/timchan0/qrmfold/blob/main/getting_started.ipynb)
notebook.
For more detail on individual functions/methods,
see below.

## Documentation
If you just want to read the documentation (most users),
click the blue **Documentation** badge at the top of this README.

Optionally, you can build the HTML docs locally:

- Install docs dependencies: `pip install -e ".[docs]"`
- Build: `cd docs && make html`
- Open: `docs/build/html/index.html`

## Release (maintainers)

This repo includes a GitHub Actions workflow that publishes to PyPI when you push a version tag like `v0.7.0`.
To use it, configure PyPI "Trusted Publishing" for this GitHub repository/project, then push a matching tag.