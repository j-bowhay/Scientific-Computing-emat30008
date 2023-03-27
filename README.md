![Test Workflow](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/test.yml/badge.svg)
![Lint Workflow](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/lint.yml/badge.svg)
[![mypy](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/mypy.yml/badge.svg)](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/mypy.yml)

# EMAT30008: Scientific Computing
Jake Bowhay: up19056

## Installation
```bash
pip install git+https://github.com/j-bowhay/Scientific-Computing-emat30008@main
```
For an editable devlopment install run:
```bash
git clone git@github.com:j-bowhay/Scientific-Computing-emat30008.git
cd Scientific-Computing-emat30008
pip install ".[dev]"
```

## Submodule Overview

### `integrate`

Solving initial value problems.

### `odes`

Provides right hand side functions for a collection of common ODEs.

### `shooting`

Numerical shooting to find limit cycles in ODEs. Also provides a number of built in phase conditions and functionality for defining arbitrary phase conditions.
