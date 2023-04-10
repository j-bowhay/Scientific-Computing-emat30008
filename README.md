# EMAT30008: Scientific Computing

Jake Bowhay: up19056

![Test Workflow](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/test.yml/badge.svg)
![Lint Workflow](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/lint.yml/badge.svg)
[![mypy](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/mypy.yml/badge.svg)](https://github.com/j-bowhay/Scientific-Computing-emat30008/actions/workflows/mypy.yml)

## Installation

```bash
pip install git+https://github.com/j-bowhay/Scientific-Computing-emat30008@main
```

For an editable development install run:

```bash
git clone git@github.com:j-bowhay/Scientific-Computing-emat30008.git
cd Scientific-Computing-emat30008
pip install -e ".[dev]"
```

## Submodule Overview

### `bvps`

Collection of convenience functions for solving common boundary value problems (BVPs).

### `continuation`

Numerical continuation methods for tracking the roots of algebraic equations, steady states
of dynamical systems and limit cycles as a parameter is varied.

### `finite_diff`

Collection of functions for finite difference approximations. Includes finites difference matrices
and boundary condition discretisations.

### `integrate`

Solving initial value problems.

### `odes`

Provides right hand side functions for a collection of common ODEs.

### `pdes`

Collection of convenience functions for solving common parabolic partial differential equations.

### `shooting`

Numerical shooting to find limit cycles in ODEs. Also provides a number of built in phase conditions and functionality for defining arbitrary phase conditions.
