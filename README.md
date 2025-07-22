# LLROpt

An experimental Julia package accompanying the paper [Tohme et al., Fast decisions with biophysically constrained gene promoter architectures](<https://arxiv.org/abs/2507.03720>).

## Description

This package implements continuous-time Markov chain (CTMC) models to study decision-making in biological molecular receptors. Warning: this is an experimental package in active development.

## Features

- Build a CTMC model from an arbitrary graph and rate matrix, specify a subset of edges as "inputs", and study the sensitivity of steady-state quantities to variations in these inputs
- Maximize an objective function over the transition rates under contraints using Lagrangian optimization
- Calculate the Fisher information of partially observed dynamics of the chain with respect to an arbitrary transition rate or set of rates

## Installation

To install the package call:
```
git clone https://github.com/ta-rekt/LLROpt.jl.git
```

then run:
```
pkg> activate LLROpt.jl
pkg> instantiate LLROpt.jl
```

## Basic usage

### Running experiments
Experiments from the paper are described in specification files located in the `spec` directory. The general command for running experiments is
```
runJL dev/{optimization_script} {experiment_specification_file} {number of CPUs} {output_directory}
```
For example, to compute the maximum Fisher information of a biological receptor with one binding site with respect to a signaling molecule's concentration, on 1 CPU, run the following command:
```
./runJL dev/specificity.opt.dev.jl 1bs_fisher_c_new.jl 1 20250722/test-experiment
```
This will create a directory `data/20250722/test-experiment` containing 1x or 2x as many JSON files as the specified number of CPUs, depending on whether one or two different target functions are optimized simultaneously per core (e.g. one of them could be a control. there will be 2 x 1 files in this example). The contents of each output file are dictionaries with parameters being varied as keys (up to 3, called beta, gamma and omega), and outputs of the optimization and other quantities of interest as a value (see specificiation file for details relative to each experiment).

### Structure of a specification file

Specification files typically have the following structure:
- Biophysical parameters
- Upper and lower bounds on optimized variables
- System (or circuit) specifications
- Initialization of optimizer and data objects
- Optimization and other parameters being varied
- Definition of objective functions to be maximized by the optimizer

### Modeling a new system
To run an experiment on a new system, add the CTMC graph describing that system into the `circuits.json` file following the same convention as the other entries. Give it a new name, and load it using that name in your specification file.

## Visualization
Check the python notebook `dev.ipynb` for visualization and further tinkering.

## Tests
Tests cases for most of the core functionality are generated in the `dev.vsnb` mathematica notebook. To run tests, simply call
```
pkg> test
```

## Contributors

This package was written by Tarek Tohme, with the great help of José Pereiro Morejón.
