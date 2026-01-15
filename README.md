# qrmfold (Quantum Reed-Muller Fold)

A Python package to generate fold-transversal gates in the quantum Reed-Muller code.

## Local Installation Instructions

### Using Conda

Create environment called `qrmfold`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate qrmfold
```

Install the `qrmfold` package in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

### Using Pip (Untested)

Install the `qrmfold` package in editable mode and its dependencies using pip:

```bash
pip install -e .
```