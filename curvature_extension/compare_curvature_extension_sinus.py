#!/usr/bin/env python3
"""Compare original DNN2GP kernel vs curvature-weighted extension on 1D Snelson data.

Important theoretical note for this setup:
- For scalar Gaussian regression, Lambda(x) = 1 / sigma_noise^2 is constant.
- Therefore phi(x) = sqrt(Lambda) * J(x) is a global scaling of J(x).
- The curvature-weighted kernel is then a constant rescaling of the original
  DNN2GP kernel (no new input-dependent curvature structure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dual_models import DualGPRegression
from laplace_models import NeuralNetworkRegression
from curvature_weighted_kernel import (
    compute_curvature_weighted_features,
    compute_curvature_weighted_kernel,
    compute_dnn2gp_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature-weighted kernel comparison on Snelson 1D regression.")
    parser.add_argument("--data-dir", type=str, default="data/snelson")
    parser.add_argument("--output-dir", type=str, default="curvature_extension_results")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sigma-noise", type=float, default=0.286)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--y-min", type=float, default=-2.7)
    parser.add_argument("--y-max", type=float, default=2.0)
    parser.add_argument("--x-min", type=float, default=-0.5)
    parser.add_argument("--x-max", type=float, default=6.5)
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=1.96,
        help="Gaussian confidence band multiplier (1.96 = 95% interval).",
    )
    parser.add_argument(
        "--band-clip-quantile",
        type=float,
        default=100.0,
        help=(
            "Optional display clipping quantile for predictive variance. "
            "Use 100 to disable clipping and show raw confidence intervals."
        ),
    )
    return parser.parse_args()


def _load_text_column(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.array([float(v) for v in path.read_text(encoding="utf-8").strip().split("\n")], dtype=np.float64)


def load_snelson_subset(data_dir: Path, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_x = _load_text_column(data_dir / "train_inputs")
    train_y = _load_text_column(data_dir / "train_outputs")
    if n_train > len(train_x):
        raise ValueError(f"Requested n_train={n_train} but only {len(train_x)} points are available.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_x))[:n_train]
    x = train_x[idx]
    y = train_y[idx]

    # Same Snelson-style gap used in existing regression_uncertainty.py.
    mask = ((x < 1.5) | (x > 3.0)).flatten()
    return x[mask][:, None], y[mask]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.double)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_snelson_subset(data_dir=data_dir, n_train=args.n_train, seed=args.seed)
    X_test = np.linspace(args.x_min, args.x_max, 1000)[:, None]

    primal_nn = NeuralNetworkRegression(
        X_train,
        y_train,
        sigma_noise=args.sigma_noise,
        delta=args.delta,
        n_epochs=args.n_epochs,
        step_size=args.step_size,
        hidden_size=args.hidden_size,
        n_layers=args.hidden_layers + 1,
        diagonal=True,
        activation=args.activation,
        lr_factor=0.99,
    )
    map_mean_test = primal_nn.predictive_map(X_test)
    map_mean_train = primal_nn.predictive_map(X_train)

    # Extract Jacobian features used by existing DNN2GP-Laplace pipeline.
    (Us_train, Ss_train), vs_train = primal_nn.UsSs("J"), primal_nn.vs("J")
    Us_test, _ = primal_nn.UsSs("J", X=X_test, y=np.ones((X_test.shape[0],)))
    y_hat = Us_train @ primal_nn.theta_star - vs_train / Ss_train

    d = primal_nn.d
    m0 = np.zeros(d)
    S0 = (1.0 / args.delta) * np.eye(d)
    s_noise = 1.0 / np.sqrt(Ss_train)

    # Original DNN2GP (existing construction).
    dual_gp_orig = DualGPRegression(Us_train, y_hat, s_noise=s_noise, m_0=m0, S_0=S0, comp_post=True)
    mean_orig, var_orig = dual_gp_orig.posterior_predictive_f(Us_test, diag_only=True)
    mean_orig_train, _ = dual_gp_orig.posterior_predictive_f(Us_train, diag_only=True)

    # Curvature-weighted extension: phi = sqrt(Lambda) J.
    # In this Gaussian regression setup Ss_train == Lambda is effectively constant.
    phi_train, lambda_scalar = compute_curvature_weighted_features(Us_train, Ss_train)
    phi_test = np.sqrt(lambda_scalar) * Us_test
    dual_gp_curv = DualGPRegression(phi_train, y_hat, s_noise=s_noise, m_0=m0, S_0=S0, comp_post=True)
    mean_curv, var_curv = dual_gp_curv.posterior_predictive_f(phi_test, diag_only=True)
    mean_curv_train, _ = dual_gp_curv.posterior_predictive_f(phi_train, diag_only=True)

    # Optional consistency check:
    # If curvature scaling is absorbed into the likelihood precision, the model is
    # just a reparameterization (for constant Lambda). Here this means unit noise.
    dual_gp_curv_noise_matched = DualGPRegression(
        phi_train,
        y_hat,
        s_noise=np.ones_like(s_noise),
        m_0=m0,
        S_0=S0,
        comp_post=True,
    )
    mean_curv_noise_matched, var_curv_noise_matched = dual_gp_curv_noise_matched.posterior_predictive_f(
        phi_test,
        diag_only=True,
    )
    var_curv_noise_matched_rescaled = var_curv_noise_matched / lambda_scalar

    # Kernel-level check: should be global scaling if Lambda is constant.
    K_orig = compute_dnn2gp_kernel(Us_train, S0)
    K_curv, _ = compute_curvature_weighted_kernel(Us_train, S0, Ss_train)
    mask = np.abs(K_orig) > 1e-10
    kernel_scale_median = float(np.median((K_curv[mask] / K_orig[mask]))) if np.any(mask) else float("nan")

    var_tot_orig = np.clip(var_orig + args.sigma_noise**2, a_min=0.0, a_max=None)
    var_tot_curv = np.clip(var_curv + args.sigma_noise**2, a_min=0.0, a_max=None)
    raw_std_orig = np.sqrt(var_tot_orig)
    raw_std_curv = np.sqrt(var_tot_curv)
    q = float(args.band_clip_quantile)
    if q >= 100.0:
        clip_orig = float(np.max(var_tot_orig))
        clip_curv = float(np.max(var_tot_curv))
        pred_std_orig = np.sqrt(var_tot_orig)
        pred_std_curv = np.sqrt(var_tot_curv)
    else:
        clip_orig = np.percentile(var_tot_orig, q)
        clip_curv = np.percentile(var_tot_curv, q)
        # Optional clipping for readability only; raw uncertainty remains in metrics.
        pred_std_orig = np.sqrt(np.clip(var_tot_orig, a_min=0.0, a_max=clip_orig))
        pred_std_curv = np.sqrt(np.clip(var_tot_curv, a_min=0.0, a_max=clip_curv))

    ci_mult = float(args.ci_multiplier)
    outside_mask = (X_test[:, 0] < float(X_train[:, 0].min())) | (X_test[:, 0] > float(X_train[:, 0].max()))
    inside_mask = ~outside_mask
    rmse_map_train = float(np.sqrt(np.mean((map_mean_train - y_train) ** 2)))
    rmse_orig_train = float(np.sqrt(np.mean((mean_orig_train - y_train) ** 2)))
    rmse_curv_train = float(np.sqrt(np.mean((mean_curv_train - y_train) ** 2)))
    mean_epistemic_std_orig = float(np.mean(np.sqrt(np.clip(var_orig, a_min=0.0, a_max=None))))
    mean_epistemic_std_curv = float(np.mean(np.sqrt(np.clip(var_curv, a_min=0.0, a_max=None))))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(X_train[:, 0], y_train, s=10, color="black", alpha=0.75, label="train points")
    # For readability we display the trained NN MAP mean (as in existing repo figure),
    # and compare uncertainty bands induced by the two kernels.
    ax.plot(X_test[:, 0], map_mean_test, color="#1f77b4", lw=2.0, label="MAP mean (shared)")
    ax.fill_between(
        X_test[:, 0],
        map_mean_test - ci_mult * pred_std_orig,
        map_mean_test + ci_mult * pred_std_orig,
        color="#1f77b4",
        alpha=0.18,
        label=f"original DNN2GP ±{ci_mult:.2f} std band",
    )
    ax.plot(X_test[:, 0], map_mean_test, color="#d62728", lw=1.7, ls="--", label="curvature extension (same mean)")
    ax.fill_between(
        X_test[:, 0],
        map_mean_test - ci_mult * pred_std_curv,
        map_mean_test + ci_mult * pred_std_curv,
        color="#d62728",
        alpha=0.16,
        label=f"curvature-weighted ±{ci_mult:.2f} std band",
    )
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Snelson 1D: original DNN2GP vs curvature-weighted extension")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    metrics_text = "\n".join(
        [
            f"train RMSE (MAP mean): {rmse_map_train:.4f}",
            f"train RMSE (orig): {rmse_orig_train:.4f}",
            f"train RMSE (curv): {rmse_curv_train:.4f}",
            f"mean epistemic std (orig): {mean_epistemic_std_orig:.4f}",
            f"mean epistemic std (curv): {mean_epistemic_std_curv:.4f}",
            f"Lambda mean/std: {float(np.mean(Ss_train)):.4f} / {float(np.std(Ss_train)):.2e}",
            f"kernel scale k_curv/k_orig: {kernel_scale_median:.4f}",
            f"CI multiplier: {ci_mult:.2f}",
            f"band clip quantile: {q:.1f}%",
        ]
    )
    ax.text(
        0.015,
        0.985,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="0.5"),
    )
    fig.tight_layout()

    fig_path = output_dir / "sinus_dnn2gp_vs_curvature_extension.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "x_test_min": float(args.x_min),
        "x_test_max": float(args.x_max),
        "x_train_min": float(X_train[:, 0].min()),
        "x_train_max": float(X_train[:, 0].max()),
        "lambda_scalar_mean": float(np.mean(Ss_train)),
        "lambda_scalar_std": float(np.std(Ss_train)),
        "is_lambda_effectively_constant": bool(np.std(Ss_train) < 1e-12),
        "kernel_scale_median_kcurv_over_korig": kernel_scale_median,
        "train_rmse_map_mean": rmse_map_train,
        "max_abs_mean_diff": float(np.max(np.abs(mean_orig - mean_curv))),
        "max_abs_std_diff": float(np.max(np.abs(pred_std_orig - pred_std_curv))),
        "max_abs_mean_diff_noise_matched": float(np.max(np.abs(mean_orig - mean_curv_noise_matched))),
        "max_abs_var_diff_noise_matched": float(np.max(np.abs(var_orig - var_curv_noise_matched))),
        "max_abs_var_diff_noise_matched_rescaled": float(
            np.max(np.abs(var_orig - var_curv_noise_matched_rescaled))
        ),
        "train_rmse_original": rmse_orig_train,
        "train_rmse_curvature": rmse_curv_train,
        "mean_epistemic_std_original": mean_epistemic_std_orig,
        "mean_epistemic_std_curvature": mean_epistemic_std_curv,
        "raw_std_mean_outside_original": float(raw_std_orig[outside_mask].mean()),
        "raw_std_mean_outside_curvature": float(raw_std_curv[outside_mask].mean()),
        "raw_std_mean_inside_original": float(raw_std_orig[inside_mask].mean()),
        "raw_std_mean_inside_curvature": float(raw_std_curv[inside_mask].mean()),
        "clipped_std_mean_outside_original": float(pred_std_orig[outside_mask].mean()),
        "clipped_std_mean_outside_curvature": float(pred_std_curv[outside_mask].mean()),
        "display_clip_quantile": q,
        "display_clip_var_original": float(clip_orig),
        "display_clip_var_curvature": float(clip_curv),
        "ci_multiplier": ci_mult,
        "note": (
            "For scalar Gaussian regression in this codepath, Lambda is constant 1/sigma_noise^2. "
            "So curvature weighting introduces a global kernel scaling but no new x-dependent structure. "
            "Any substantial predictive change comes from changed kernel-vs-noise scaling. "
            "In the noise-matched check, mean becomes numerically identical and variance differs by the same "
            "global Lambda factor."
        ),
    }
    (output_dir / "curvature_extension_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {output_dir / 'curvature_extension_metrics.json'}")
    print("Lambda mean/std:", metrics["lambda_scalar_mean"], metrics["lambda_scalar_std"])
    print("Kernel scale median (k_curv / k_orig):", metrics["kernel_scale_median_kcurv_over_korig"])


if __name__ == "__main__":
    main()
