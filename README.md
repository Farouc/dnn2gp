# dnn2gp

Bayesian machine learning project with implementations of Laplace, dual, and variational models.

## Reproduce Experiments

All commands below assume you are in the project root and using the `base` environment.
If needed, replace `python` with `/home/farouc/envs/base/bin/python`.

### 1) Snelson uncertainty (paper-like)

```bash
python regression_uncertainty.py \
  --format png \
  --output-dir figures \
  --laplace-epochs 20000 \
  --vi-epochs 16000 \
  --laplace-pred-samples 1000 \
  --vi-pred-samples 1000 \
  --gp-restarts 20 \
  --seed 100
```

Outputs:
- `figures/regression_uncertainty_laplace.png`
- `figures/regression_uncertainty_vi.png`
- `figures/snelson_data.png`

### 2) Snelson playground (easy method comparison)

```bash
python snelson_experiment_playground.py --format png --output-dir figures
```

Output:
- `figures/snelson_method_comparison.png`

### 3) Kernel + predictive arrays (MNIST/CIFAR, from saved checkpoints)

```bash
python kernel_and_predictive.py
```

Outputs in `results/` (examples):
- `CIFAR_Laplace_kernel.npy`
- `CIFAR_Laplace_predictive_var_f.npy` (epistemic)
- `CIFAR_Laplace_predictive_noise.npy` (aleatoric)
- `CIFAR_VI_kernel.npy`

### 4) Kernel and uncertainty plots from `results/`

Laplace CIFAR plots:

```bash
python kernel_and_predictive_plots.py --prefix CIFAR_Laplace --format png --output-dir figures
```

VI CIFAR plots:

```bash
python kernel_and_predictive_plots.py --prefix CIFAR_VI --format png --output-dir figures
```

Key outputs:
- `figures/CIFAR_Laplace_kernel.png`
- `figures/CIFAR_Laplace_pred_mean_ste.png`
- `figures/CIFAR_Laplace_var_f.png` (epistemic)
- `figures/CIFAR_Laplace_var_y.png` (aleatoric)
- `figures/CIFAR_VI_kernel.png`

### 5) Hyperparameter tuning (marglik, toy)

Fast reproducible run:

```bash
python marglik.py \
  --name repro_fast \
  --mode all \
  --n_retries 1 \
  --n_params 21 \
  --width_n_params 20 \
  --width_log_max 2.3 \
  --n_epochs 100 \
  --vi_epochs 100 \
  --n_processes 1 \
  --seed 7
```

Result files:
- `results/reg_ms_delta_repro_fast.pkl`
- `results/reg_ms_width_repro_fast.pkl`

Plot toy-paper figures from those results:

```bash
python marglik_plots.py --name repro_fast --mode toy
python marglik_plots.py --name repro_fast --mode toy_extra
```

Outputs:
- `figures/marglik_delta_toy_laplace.pdf`
- `figures/marglik_delta_toy_VI.pdf`
- `figures/marglik_delta_toy_fits.pdf`
- `figures/marglik_width_toy_laplace.pdf`
