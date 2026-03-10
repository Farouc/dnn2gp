# Bayesian Machine Learning Course Project

This repository contains the project work for the **Bayesian Machine Learning** course.

Authors:
- **Farouk Yartaoui**
- **Rida Assalouh**
- **El Mehdi Nezahi**

## Overview

The project impelemnts the paper "Approximate Inference Turns Neural Networks to Gaussian Processes", while doing some the following:
- Laplace / DNN2GP experiments
- Variational baselines (VOGN / OGGN)
- Marginal-likelihood hyperparameter sweeps
- Robustness and uncertainty analyses on MNIST
- Curvature-based kernel extensions

All commands below are expected to be run from the repository root.

## Repository Structure

```text
.
├── src/                          # Core models and utilities
├── experiments/                  # Runnable experiments grouped by topic
│   ├── dnn2gp/
│   ├── regression/
│   ├── mnist/
│   ├── vogn/
│   ├── marglik/
│   ├── marglik_delta_gpu/
│   ├── curvature_hardband_2d/
│   └── curvature_extension/
├── results/                      # Saved arrays, metrics, and experiment outputs
├── figures/                      # Saved publication/demo figures
├── models/                       # Checkpoints
├── data/                         # Datasets (downloaded/generated)
└── experiments_results_summary.ipynb        # Unified demo notebook
```

## Installation

1. Create and activate an environment (example with `venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional, GPU) install CUDA-enabled PyTorch matching your CUDA driver from the official PyTorch instructions.

## Quick Demo

Open the unified notebook:

```bash
jupyter notebook experiments_results_summary.ipynb
```

## Run Experiments

### 1) DNN2GP kernels and predictive quantities

```bash
python experiments/dnn2gp/kernel_and_predictive.py
python experiments/dnn2gp/kernel_and_predictive_plots.py --prefix CIFAR_Laplace --output-dir figures
python experiments/dnn2gp/kernel_and_predictive_plots.py --prefix CIFAR_VI --output-dir figures
python experiments/dnn2gp/reproduce_mnist_dnn2gp_laplace.py
```

Outputs are written under `results/` and `figures/`.

### 2) Regression uncertainty (Snelson) and scaling

```bash
python experiments/regression/regression_uncertainty.py --output-dir figures --format png
python experiments/regression/snelson_experiment_playground.py --output-dir figures --format png
python experiments/regression/laplace_variance_scaling_experiment.py --output-dir figures --results-dir results
```

### 3) MNIST robustness / uncertainty experiments

```bash
python experiments/mnist/mnist_adversarial_uncertainty_experiment.py --device auto --output-dir results
python experiments/mnist/mnist_transformation_uncertainty_experiment.py --device auto --output-dir results
python experiments/mnist/mnist_3_8_interpolation_uncertainty_experiment.py --device auto --output-dir results
python experiments/mnist/mnist_adversarial_epsilon_sweep_experiment.py --device auto --output-dir results/adversarial_examples
python experiments/mnist/mnist_adversarial_gp_vs_dnn_confidence_experiment.py --device auto --output-dir results/adversarial_examples
python experiments/mnist/mnist_adversarial_transferability_experiment.py --device auto --output-dir results/adversarial_examples
python experiments/mnist/mnist_iuc_experiment.py --device auto --output-dir results/adversarial_examples
python experiments/mnist/mnist_label_noise_reliability_experiment.py --device auto --output-dir results/adversarial_examples
python experiments/mnist/mnist_selective_classification_experiment.py --device auto --results-dir results --figures-dir figures
```

### 4) Curvature hard-band 2D experiment

```bash
python experiments/curvature_hardband_2d/run_curvature_hardband_2d.py --device auto
python experiments/curvature_hardband_2d/plot_kernel_region_ordered.py --results-dir results/curvature_hardband_2d
python experiments/curvature_hardband_2d/plot_kernel_region_binned.py --results-dir results/curvature_hardband_2d
```

### 5) Curvature extension experiments

```bash
python experiments/curvature_extension/compare_curvature_extension_sinus.py
python experiments/curvature_extension/compare_curvature_extension_mnist_small.py --device auto
python experiments/curvature_extension/compare_curvature_extension_fashion_mnist_small.py --device auto
```

Primary outputs: `results/curvature_extension/`.

### 6) Marginal likelihood sweeps

Toy sweeps:

```bash
python experiments/marglik/marglik.py --name repro_fast --mode all --n_retries 1 --n_params 21 --width_n_params 20 --n_epochs 100 --vi_epochs 100 --n_processes 1
python experiments/marglik/marglik_plots.py --name repro_fast --mode toy
python experiments/marglik/marglik_plots.py --name repro_fast --mode toy_extra
```

Delta sweeps (GPU-aware scripts):

```bash
python experiments/marglik_delta_gpu/run_marglik_delta_gpu.py --device auto
python experiments/marglik_delta_gpu/run_marglik_delta_real.py --device auto
```

### 7) VOGN / OGGN experiment suite

```bash
python experiments/vogn/run_all_vogn_experiments.py --device auto
python experiments/vogn/vogn_small_datasets_experiments.py --output-dir figures --results-dir results
python experiments/vogn/vogn_variance_scaling_experiment.py --output-dir figures --results-dir results
```

Primary outputs: `results/vogn/`, `results/`, `figures/`.

## Notes

- `--device auto` uses CUDA when available, otherwise CPU.
- Most scripts create output directories automatically.
- Some optional scripts (`experiments/marglik/marglik_uci.py`) depend on `vadam` and additional UCI data setup.
