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

Paper-scale (very long on CPU):

```bash
python marglik.py \
  --name repro_full \
  --mode all \
  --n_retries 10 \
  --n_params 21 \
  --width_n_params 30 \
  --width_log_max 3.0 \
  --n_epochs 10000 \
  --vi_epochs 5000 \
  --n_processes 0 \
  --seed 7
```

## Results Folder Inventory (March 8, 2026)

This section lists the experiments that already have artifacts in `results/`.

### A) DNN2GP kernel/predictive arrays (MNIST/CIFAR)

Generating script:

```bash
python kernel_and_predictive.py
```

Current files:
- `results/BIN_MNIST_VI_gp_predictive_mean.npy` (`300 x 2`)
- `results/BIN_MNIST_VI_kernel.npy` (`300 x 300`)
- `results/MNIST_Laplace_gp_predictive_mean.npy` (`300 x 10`)
- `results/MNIST_Laplace_predictive_mean.npy` (`300 x 10`)
- `results/MNIST_Laplace_predictive_var_f.npy` (`300 x 10`, epistemic)
- `results/MNIST_Laplace_predictive_noise.npy` (`300 x 10`, aleatoric)
- `results/MNIST_Laplace_kernel.npy` (`300 x 300`)
- `results/MNIST_VI_gp_predictive_mean.npy` (`300 x 10`)
- `results/MNIST_VI_kernel.npy` (`300 x 300`)
- `results/CIFAR_Laplace_gp_predictive_mean.npy` (`300 x 10`)
- `results/CIFAR_Laplace_predictive_mean.npy` (`300 x 10`)
- `results/CIFAR_Laplace_predictive_var_f.npy` (`300 x 10`, epistemic)
- `results/CIFAR_Laplace_predictive_noise.npy` (`300 x 10`, aleatoric)
- `results/CIFAR_Laplace_kernel.npy` (`300 x 300`)
- `results/CIFAR_VI_gp_predictive_mean.npy` (`300 x 10`)
- `results/CIFAR_VI_kernel.npy` (`300 x 300`)

### B) Marglik hyperparameter tuning (toy regression)

Generating script:

```bash
python marglik.py --name <run_name> --mode all ...
```

Current files:
- `results/reg_ms_delta_smoke_marglik.pkl` (smoke delta sweep, 2 parameter values)
- `results/reg_ms_width_smoke_marglik.pkl` (smoke width sweep, 2 parameter values)
- `results/reg_ms_delta_repro_fast.pkl` (repro fast delta sweep, 21 parameter values)
- `results/reg_ms_width_repro_fast.pkl` (repro fast width sweep, 18 parameter values)

Each PKL contains:
- `datasets`: sampled train/test toy datasets
- `params`: sweep values
- `results`: metrics per parameter (marginal likelihoods, losses, fitted curves)

### C) VOGN small-dataset regression comparison

Generating script:

```bash
python vogn_small_datasets_experiments.py --output-dir figures --results-dir results
```

Current file:
- `results/vogn_small_datasets_metrics.json`

Contains per-dataset metrics for `gap_sine`, `hetero_sine`, and `cubic`:
- training size, noise level
- mean/median predictive variance

### D) Scaling experiment (posterior variance vs dataset size, non-VOGN)

Generating script:

```bash
python laplace_variance_scaling_experiment.py --output-dir figures --results-dir results
```

Current file:
- `results/laplace_scaling_metrics.csv`

Columns:
- `scale`, `n_train`, `method`, `epochs`
- `mean_var_all`, `mean_var_center`, `std_var_all`

Methods in this CSV:
- `laplace_j`
- `dual_blr_diag`

## MNIST Adversarial / Robustness Uncertainty (Detailed)

This section explains the 3 new MNIST experiments:
- adversarial perturbations,
- geometric transformations (rotation/translation),
- interpolation between digit `3` and digit `8`.

All three experiments use the same Laplace dnn2gp uncertainty pipeline.

### Common Pipeline Used in All 3 Experiments

Scripts:
- `mnist_adversarial_uncertainty_experiment.py`
- `mnist_transformation_uncertainty_experiment.py`
- `mnist_3_8_interpolation_uncertainty_experiment.py`
- shared utilities in `mnist_dnn2gp_experiment_utils.py`

Model/checkpoint:
- `LeNet5` 10-class MNIST classifier loaded from `models/full_mnist_lenet_adaml2.tk`.

Laplace posterior precision:
- A diagonal Laplace precision vector is computed once (or loaded from cache):
  - cache file: `results/mnist_laplace_post_prec_2000.pt`
  - computed on a random MNIST train subset of size `2000`
  - prior precision: `1e-4`
  - batch size: `64`

Per-image uncertainty computation:
1. Compute logits `f(x)` and softmax probabilities `p = softmax(f(x))`.
2. Build the softmax local covariance factor:
   - `Lambda = diag(p) - p p^T`
3. Compute Jacobian of each class logit wrt parameters:
   - `J_k = grad_theta f_k(x)` for `k=0..9`
4. Use diagonal Laplace covariance `diag(1 / post_prec)` to form logit covariance:
   - `Cov_f = J diag(1/post_prec) J^T`
5. Compute class-wise epistemic variance in probability space:
   - `var_f = diag(Lambda * Cov_f * Lambda)`
6. Compute class-wise aleatoric term:
   - `noise = p - p^2`

Stored per sample:
- `probs`, `pred`, `max_prob`
- `entropy = -sum_c p_c log p_c`
- `epistemic` (class-wise `var_f`)
- `aleatoric` (class-wise `p_c(1-p_c)`)

### Why Some Uncertainty Numbers Are Very Large (and Not in [0, 1])

Important point:
- Not every uncertainty metric is a probability.

Bounded metrics:
- `max_prob` is in `[0, 1]`.
- `entropy` is in `[0, log(10)]` for MNIST 10 classes, so `[0, ~2.3026]`.
- each class-wise aleatoric term `p_c(1-p_c)` is in `[0, 0.25]`.
- sum of aleatoric over 10 classes is at most `2.5`.

Unbounded metric:
- `epistemic` from Laplace dnn2gp is a variance-like quantity derived from Jacobians and inverse precision.
- It is not normalized to `[0, 1]`.
- It can become very large when:
  - Jacobian norms are large (input is off-manifold / adversarial / heavily transformed),
  - posterior precision is small in influential parameter directions (`1/post_prec` large),
  - class-wise variances are summed across classes.

So large epistemic values indicate strong model uncertainty in parameter-induced predictive variation, not invalid math.

### 1) Adversarial Experiment (FGSM)

Script:

```bash
python mnist_adversarial_uncertainty_experiment.py --device auto --output-dir results
```

Outputs:
- `results/mnist_adversarial_uncertainty.png`
- `results/mnist_adversarial_uncertainty_metrics.csv`

What was done:
- Build a balanced test subset (`n_per_class=3`, so 30 images total).
- Generate FGSM adversarial images with `epsilon=0.25`:
  - `x_adv = clip(x + epsilon * sign(grad_x CE(model(x), y)), 0, 1)`
- Compute uncertainties for both clean and adversarial images.

What the figure shows:
- Top-left: clean image strip.
- Top-right: corresponding adversarial strip.
- Bottom-left: scatter of `alea_sum` vs `epi_sum` (log-log), clean vs adversarial.
  - adversarial points shift to much higher epistemic region.
- Bottom-right: mean metrics for clean vs adversarial:
  - entropy,
  - epistemic sum,
  - aleatoric sum,
  - `1 - max_prob`.

CSV columns explained:
- `idx`: sample index in the selected subset.
- `true_label`: ground-truth class.
- `pred_clean`, `pred_adv`: predicted class before/after FGSM.
- `attack_success`: `1` if adversarial prediction differs from true label.
- `entropy_clean`, `entropy_adv`: predictive entropy.
- `epi_clean`, `epi_adv`: summed epistemic uncertainty across classes.
- `alea_clean`, `alea_adv`: summed aleatoric uncertainty across classes.
- `maxprob_clean`, `maxprob_adv`: max softmax confidence.

### 2) Transformation Experiment (Rotations + Translations)

Script:

```bash
python mnist_transformation_uncertainty_experiment.py --device auto --output-dir results
```

Outputs:
- `results/mnist_transformation_uncertainty.png`
- `results/mnist_transformation_uncertainty_metrics.csv`

What was done:
- Select a balanced base subset (`n_per_class=1`, 10 images total).
- Generate transformed sets:
  - rotations: angles from `-60` to `+60` (13 values),
  - horizontal translations: from `-8` to `+8` pixels (step `2`).
- For each transformed image, compute uncertainty metrics.
- Aggregate by transform parameter (mean across base samples).

What the figure shows:
- Left panels: representative digit under rotations/translations.
- Top-right: uncertainty vs rotation angle (log y-scale for epi/alea), plus max probability on secondary axis.
- Bottom-right: uncertainty vs translation shift, plus max probability.
- Typical pattern: uncertainty is lowest near mild/center transforms and increases for strong distortions.

CSV columns explained:
- `transform_type`: `rotation` or `translation`.
- `param_value`: angle (degrees) or shift (pixels).
- `sample_id`: which base image.
- `true_label`, `pred`, `entropy`, `epi_sum`, `alea_sum`, `max_prob`.

### 3) Interpolation Experiment (3 <-> 8)

Script:

```bash
python mnist_3_8_interpolation_uncertainty_experiment.py --device auto --output-dir results
```

Outputs:
- `results/mnist_3_8_interpolation_uncertainty.png`
- `results/mnist_3_8_interpolation_uncertainty_metrics.csv`

What was done:
- Sample `n_pairs=6` real `3` images and `6` real `8` images.
- For each pair, build linear interpolation path with `alpha in [0,1]` (21 values):
  - `x(alpha) = (1-alpha) * x3 + alpha * x8`
- Compute uncertainties and class probabilities along each path.

What the figure shows:
- Top-left: one example interpolation strip (`pair 0`).
- Top-right: `P(class=3)` and `P(class=8)` vs alpha for `pair 0`.
- Bottom-left: mean uncertainty curves vs alpha (epistemic, aleatoric, entropy).
- Bottom-right: heatmap of predicted class along interpolation for all pairs.
  - shows where the model switches from `3` to `8`.

CSV columns explained:
- `pair_id`, `alpha`, `pred`
- `prob_3`, `prob_8`
- `entropy`, `epi_sum`, `alea_sum`, `max_prob`

### Practical Interpretation Notes

- Use entropy/max-prob for confidence-style interpretation.
- Use epistemic for model uncertainty about unseen/shifted inputs.
- Very high epistemic spikes under attacks or strong transforms usually indicate distribution shift sensitivity.
- Compare clean vs perturbed relatively (change/trend), not by expecting a fixed absolute scale like `[0,1]` for epistemic.
