# Curvature-Weighted DNN2GP Extension (1D Gaussian Regression)

This folder contains a minimal extension on top of the existing `src/` pipeline.

## What is implemented

- `curvature_weighted_kernel.py`
  - `compute_dnn2gp_kernel(...)`
  - `compute_curvature_weighted_features(...)`
  - `compute_curvature_weighted_kernel(...)`
- `compare_curvature_extension_sinus.py`
  - runs the existing Snelson-style 1D Laplace regression setup
  - computes original DNN2GP prediction and curvature-weighted prediction
  - saves one comparison figure and one metrics JSON

## Theoretical note for this setup

For scalar-output Gaussian regression used here:

- `Lambda(x) = 1 / sigma_noise^2` (constant)
- `phi(x) = sqrt(Lambda) J(x)` is a global scaling of `J(x)`
- `k_curv(x, x')` is therefore a global scaling of the original DNN2GP kernel

So this extension is *structurally trivial* on this specific setup:
it does not add input-dependent curvature shape beyond a kernel scale factor.

## Run

From project root:

```bash
python experiments/curvature_extension/compare_curvature_extension_sinus.py
```

Outputs go to:

- `results/curvature_extension/sinus_dnn2gp_vs_curvature_extension.png`
- `results/curvature_extension/curvature_extension_metrics.json`

## Small MNIST Classification Extension

Run:

```bash
python experiments/curvature_extension/compare_curvature_extension_mnist_small.py
```

Outputs:

- `results/curvature_extension/mnist_curvature_extension/mnist_small_dnn2gp_vs_curvature_extension.png`
- `results/curvature_extension/mnist_curvature_extension/mnist_small_curvature_extension_metrics.json`
- kernel arrays (`.npy`) and prediction arrays (`.npz`) in the same folder

This experiment is intentionally lightweight (small train/test subsets, limited MC samples) to avoid heavy GPU usage.
