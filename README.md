# qrmfold (Quantum Reed-Muller Fold)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://timchan0.github.io/qrmfold/)

A Python package to construct addressable gates
for self-dual quantum Reed-Muller codes
as a sequence of transversal and fold-transversal gates.
Click [here](https://arxiv.org/abs/2602.09788)
for the accompanying paper.

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

## Release (maintainers)
This repo includes a GitHub Actions workflow that publishes to PyPI when a version tag like `v0.7.0` is pushed.