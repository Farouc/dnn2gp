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

## VOGN / OGGN Training-to-GP Experiments (Detailed)

This section documents the full experiment suite generated in `voggn_results/` by:

```bash
source ~/envs/base/bin/activate
python run_all_vogn_experiments.py --device cuda
```

Code organization:
- `run_all_vogn_experiments.py` orchestrates all experiments.
- `experiments_vogn/` contains modular training, GP prediction, MC prediction, kernel, uncertainty, and plotting utilities.
- In this codebase, "OGGN" experiments are run with the `VOGGN` optimizer class from `src/vogn.py`.

All experiments save:
- plots (`.png`)
- arrays (`.npz` / `.npy`)
- metrics (`metrics.json`)

Global summary:
- `voggn_results/summary_metrics.json`

### Experiment 1: Evolution of GP Predictive Distribution During Training

Folder:
- `voggn_results/experiment1_gp_evolution/`

Setup:
- 1D regression toy dataset with a central gap: `x in [-4,4]`, removed interval `[-1,1]`.
- Target: `y = sin(x) + noise`.
- Small MLP (`hidden=32`, two hidden layers, `tanh`).
- Checkpoints: `step = [0, 10, 50, 100, 200, 500]`.
- For each checkpoint and optimizer (VOGN, OGGN), compute linearized GP mean/variance and plot:
  - train points
  - predictive mean curve
  - `±2 sigma` band

Saved files:
- `vogn/mean_variance_step_XXX.png`, `oggn/mean_variance_step_XXX.png`
- `vogn/gp_step_XXX.npz`, `oggn/gp_step_XXX.npz`

Observed results (from `summary_metrics.json`):
- Mean predictive variance (`mean_var`) generally decreases during training for both methods.
- VOGN: `0.558 (step 0) -> 0.233 (step 200)`, then rises at `step 500` (`0.435`).
- OGGN: `0.512 (step 0) -> 0.190 (step 200)`, remains lower at `step 500` (`0.206`).

Interpretation:
- Both methods reduce posterior uncertainty as they fit data.
- OGGN shows a more stable low-variance endpoint in this run.
- VOGN shows late-stage variance increase, indicating a less stable final posterior geometry for this seed/hyperparameter choice.

### Experiment 2: Optimization Behavior Comparison

Folder:
- `voggn_results/experiment2_optimizer_comparison/`

Compared optimizers:
- `VOGN`, `OGGN`, `Adam`, `RMSprop`.

Tasks:
- Toy regression (same gap-sine setup).
- Binary MNIST classification (`0 vs 1`).

Saved files:
- Regression curves: `regression_loss_vs_iteration.png`, `regression_test_mse_vs_iteration.png`
- Classification curves: `classification_loss_vs_iteration.png`, `classification_accuracy_vs_iteration.png`
- Arrays: `regression_histories.npz`, `classification_histories.npz`
- Metrics: `metrics.json`

Observed results:
- Regression final test MSE:
  - `VOGN: 0.2718`
  - `OGGN: 0.2806`
  - `Adam: 0.0030`
  - `RMSprop: 0.0159`
- Binary MNIST final test accuracy:
  - `VOGN: 0.999`
  - `OGGN: 1.000`
  - `Adam: 1.000`
  - `RMSprop: 1.000`

Interpretation:
- On small binary MNIST, all methods solve the task almost perfectly.
- On this toy regression setting, deterministic optimizers reached lower test MSE than variational optimizers under current hyperparameters.
- This does not invalidate VOGN/OGGN; it reflects that variational training is optimizing a different objective and is sensitive to prior/precision settings.

### Experiment 3: GP Predictive Distribution vs Monte Carlo Weight Sampling

Folder:
- `voggn_results/experiment3_gp_vs_mc/`

Goal:
- Test how closely linearized GP predictions match nonlinear Monte Carlo predictions from sampled weights `w ~ N(mu_t, Sigma_t)`.

Method:
- Reuse checkpoints from Experiment 1.
- For each checkpoint and optimizer:
  - GP predictive mean/variance from Jacobians.
  - MC mean/variance from ~80 sampled networks.
  - Compute mismatch metrics: `mean_mse`, `var_mse`.

Saved files:
- Comparison figures: `*_gp_vs_mc_step_XXX.png`
- Arrays: `*_step_XXX.npz`
- Metrics: `metrics.json`

Observed results:
- OGGN has consistently low mismatch at late checkpoints:
  - `var_mse` reaches `0.0054` (step 500).
- VOGN variance mismatch is also low at step 500 (`0.0092`), but mean mismatch increases (`mean_mse = 0.3612`).

Interpretation:
- Variance agreement is reasonably good for both methods at late stages.
- Mean agreement is more stable for OGGN in this run.
- The figures are useful diagnostics of linearization quality across training.

### Experiment 4: Kernel Evolution During Training

Folder:
- `voggn_results/experiment4_kernel_evolution/`

Kernels plotted at steps `[0, 10, 50, 100, 200]`:
- NTK-like: `J J^T`
- VOGN kernel: `J Sigma_t J^T`
- OGGN kernel: `J Sigma_hat_t J^T`

Saved files:
- Heatmaps: `kernel_ntk_step_XXX.png`, `kernel_vogn_step_XXX.png`, `kernel_oggn_step_XXX.png`
- Matrices: corresponding `.npy` files
- Metrics: `metrics.json` (including trace values)

Observed results:
- NTK trace remains much larger than posterior-weighted kernels (expected due to no posterior covariance scaling).
- Trace trends from step 0 to 200:
  - VOGN kernel trace: `44.76 -> 18.75`
  - OGGN kernel trace: `41.09 -> 15.27`

Interpretation:
- Posterior-aware kernels contract during training, indicating reduced uncertainty-weighted function-space variability.
- OGGN contracts slightly more by step 200 in this run.

### Experiment 5: OOD Detection with Uncertainty

Folder:
- `voggn_results/experiment5_ood_detection/`

Protocol:
- Train uncertainty model on MNIST digits `{0,1}`.
- Evaluate on a subset of full MNIST test `{0..9}`.
- Define OOD as digits not in `{0,1}`.
- Compute per-sample:
  - predictive entropy
  - predictive variance proxy
- Measure AUROC for OOD detection.

Saved files:
- Histograms: `vogn_ood_entropy.png`, `vogn_ood_variance.png`, `oggn_ood_entropy.png`, `oggn_ood_variance.png`
- Arrays: `vogn_ood_arrays.npz`, `oggn_ood_arrays.npz`
- Metrics: `metrics.json`

Observed results:
- Entropy AUROC:
  - `VOGN: 0.9261`
  - `OGGN: 0.9542`
- Predictive variance AUROC:
  - `VOGN: 0.9212`
  - `OGGN: 0.9489`
- Seen vs unseen separation (entropy means):
  - VOGN: `0.0102 (seen)` vs `0.2484 (unseen)`
  - OGGN: `0.0140 (seen)` vs `0.3044 (unseen)`

Interpretation:
- Both models provide useful uncertainty for OOD separation.
- OGGN gave stronger AUROC on this run.
- Histogram overlap is lower for OGGN, consistent with better separation.

### Experiment 6: Calibration During Training

Folder:
- `voggn_results/experiment6_calibration/`

Compared methods:
- `VOGN`, `OGGN`, `Adam` on full MNIST (`0..9`, subset sizes for tractability).

Metrics tracked by epoch:
- Negative log-likelihood (NLL)
- Expected calibration error (ECE)
- Brier score

Saved files:
- Metric curves: `nll_vs_epoch.png`, `ece_vs_epoch.png`, `brier_vs_epoch.png`
- Reliability diagrams: `calibration_curve_vogn.png`, `calibration_curve_oggn.png`, `calibration_curve_adam.png`
- Arrays: `calibration_arrays.npz`
- Metrics: `metrics.json`

Final-epoch results:
- NLL:
  - `VOGN: 1.0083`
  - `OGGN: 1.0167`
  - `Adam: 0.3790`
- ECE:
  - `VOGN: 0.0781`
  - `OGGN: 0.0750`
  - `Adam: 0.0475`
- Brier:
  - `VOGN: 0.4670`
  - `OGGN: 0.4902`
  - `Adam: 0.1622`

Interpretation:
- All methods improve substantially during training.
- In this run, Adam ends with stronger calibration/probability scores.
- VOGN and OGGN remain competitive in ECE while carrying posterior uncertainty structure useful for GP-style analyses and OOD behavior.

### Reproducibility Notes

- The suite uses small datasets/subsets to keep Jacobian and kernel computations tractable.
- Results are seed- and hyperparameter-dependent; compare methods over multiple seeds before drawing final conclusions.
- Use `voggn_results/summary_metrics.json` for quick numeric comparison and the corresponding PNGs for qualitative diagnosis.
