import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laplace_models import NeuralNetworkRegression
from dual_models import DualLinearRegression


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sine_dataset(n: int, sigma_noise: float, x_low: float = -4.0, x_high: float = 4.0):
    x = np.sort(np.random.uniform(x_low, x_high, size=n))
    y_true = np.sin(1.4 * x) + 0.15 * np.cos(0.7 * x)
    y = y_true + np.random.normal(0.0, sigma_noise, size=n)
    return x[:, None], y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Laplace scaling experiment: dataset sizes x10/x100/x1000 and posterior variance trend."
    )
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--base-n", type=int, default=10, help="Base dataset size; scales produce N,10N,100N,1000N.")
    parser.add_argument("--sigma-noise", type=float, default=0.20)
    parser.add_argument("--test-points", type=int, default=600)
    parser.add_argument("--laplace-pred-samples", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.1, help="Prior precision.")
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--lr-factor", type=float, default=0.99)
    parser.add_argument("--quick", action="store_true", help="Fast smoke mode for wiring checks.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.double)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scales = [1, 10, 100, 1000]
    ns = [args.base_n * s for s in scales]
    if args.quick:
        epochs_by_scale = {1: 300, 10: 220, 100: 140, 1000: 80}
        pred_samples = min(args.laplace_pred_samples, 120)
    else:
        epochs_by_scale = {1: 2400, 10: 1400, 100: 700, 1000: 260}
        pred_samples = args.laplace_pred_samples

    x_test = np.linspace(-4.2, 4.2, args.test_points)[:, None]
    y_true_test = np.sin(1.4 * x_test[:, 0]) + 0.15 * np.cos(0.7 * x_test[:, 0])
    center_mask = (x_test[:, 0] >= -2.0) & (x_test[:, 0] <= 2.0)

    methods = ["laplace_j", "dual_blr_diag"]
    method_labels = {
        "laplace_j": "Laplace (J)",
        "dual_blr_diag": "Dual-BLR diag",
    }
    method_colors = {
        "laplace_j": "#d62728",
        "dual_blr_diag": "#1f77b4",
    }

    predictions = {name: [] for name in methods}
    rows = ["scale,n_train,method,epochs,mean_var_all,mean_var_center,std_var_all"]

    for idx, (scale, n_train) in enumerate(zip(scales, ns)):
        print(f"\n=== Scale x{scale} (n_train={n_train}) ===")
        x_train, y_train = make_sine_dataset(n=n_train, sigma_noise=args.sigma_noise)
        n_epochs = epochs_by_scale[scale]

        model = NeuralNetworkRegression(
            x_train,
            y_train,
            sigma_noise=args.sigma_noise,
            delta=args.delta,
            n_epochs=n_epochs,
            step_size=args.step_size,
            hidden_size=args.hidden_size,
            n_layers=args.hidden_layers + 1,
            diagonal=True,
            activation="tanh",
            lr_factor=args.lr_factor,
            seed=args.seed + idx,
        )

        # Method 1: direct Laplace predictive with Jacobian linearization.
        laplace_mean, laplace_var = model.posterior_predictive_f(
            x_test,
            "J",
            n_samples=pred_samples,
            compute_cov=True,
            diag_only=True,
        )
        laplace_var = np.clip(laplace_var.flatten(), 1e-12, None)
        laplace_mean = laplace_mean.flatten()

        # Method 2: dual Bayesian linear regression with diagonal posterior precision.
        m_0 = np.zeros(model.d)
        S_0 = (1.0 / args.delta) * np.eye(model.d)
        (Us, Ss), vs = model.UsSs("J"), model.vs("J")
        X_hat = Us
        y_hat = Us @ model.theta_star - vs / Ss
        s_noise = 1.0 / np.sqrt(Ss)
        X_hat_test, _ = model.UsSs("J", X=x_test, y=np.ones((x_test.shape[0],)))

        dual_blr = DualLinearRegression(X_hat, y_hat, s_noise, m_0, S_0=S_0)
        dual_blr.P_post = np.diag(np.diag(dual_blr.P_post))
        dual_blr.S_post = np.diag(1.0 / np.diag(dual_blr.P_post))
        dual_mean, dual_var = dual_blr.posterior_predictive_f(X_hat_test, diag_only=True)
        dual_var = np.clip(dual_var.flatten(), 1e-12, None)
        dual_mean = dual_mean.flatten()

        outputs = {
            "laplace_j": (laplace_mean, laplace_var),
            "dual_blr_diag": (dual_mean, dual_var),
        }
        for method in methods:
            mean, var = outputs[method]
            predictions[method].append(
                {
                    "scale": scale,
                    "n_train": n_train,
                    "x_train": x_train,
                    "y_train": y_train,
                    "mean": mean,
                    "var": var,
                }
            )

            mean_var_all = float(np.mean(var))
            mean_var_center = float(np.mean(var[center_mask]))
            std_var_all = float(np.std(var))
            rows.append(
                f"{scale},{n_train},{method},{n_epochs},{mean_var_all:.8f},{mean_var_center:.8f},{std_var_all:.8f}"
            )
            print(
                f"  {method_labels[method]}: "
                f"mean_var_all={mean_var_all:.6f}, mean_var_center={mean_var_center:.6f}"
            )

    n_rows = len(methods)
    n_cols = len(scales)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.9 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = axs[:, None]

    for r, method in enumerate(methods):
        for c, pred in enumerate(predictions[method]):
            ax = axs[r, c]
            x_train = pred["x_train"]
            y_train = pred["y_train"]
            mean = pred["mean"]
            var = pred["var"]

            std_y = np.sqrt(np.clip(var + args.sigma_noise**2, 1e-12, None))
            stride = max(1, pred["n_train"] // 500)
            ax.scatter(x_train[::stride, 0], y_train[::stride], s=8, c="black", alpha=0.48)
            ax.plot(x_test[:, 0], y_true_test, color="gray", linestyle="--", linewidth=1.2)
            ax.plot(x_test[:, 0], mean, color=method_colors[method], linewidth=1.8)
            ax.fill_between(
                x_test[:, 0],
                mean - 2.0 * std_y,
                mean + 2.0 * std_y,
                alpha=0.20,
                color=method_colors[method],
            )
            if r == 0:
                ax.set_title(f"x{pred['scale']} (N={pred['n_train']})")
            if c == 0:
                ax.set_ylabel(method_labels[method])
            ax.grid(alpha=0.2)

    axs[-1, 0].set_xlabel("x")
    if n_cols > 1:
        axs[-1, 1].set_xlabel("x")
    if n_cols > 2:
        axs[-1, 2].set_xlabel("x")
    if n_cols > 3:
        axs[-1, 3].set_xlabel("x")
    fig.suptitle("Laplace scaling: predictive mean and uncertainty", y=1.01)
    fig.tight_layout()
    pred_path = output_dir / "laplace_scaling_predictive.png"
    fig.savefig(pred_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {pred_path}")

    fig2, (ax_all, ax_center) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    metrics = {m: {"all": [], "center": []} for m in methods}
    for row in rows[1:]:
        scale_s, _, method, _, mean_all_s, mean_center_s, _ = row.split(",")
        metrics[method]["all"].append((int(scale_s), float(mean_all_s)))
        metrics[method]["center"].append((int(scale_s), float(mean_center_s)))

    for method in methods:
        xs_all = [x for x, _ in metrics[method]["all"]]
        ys_all = [y for _, y in metrics[method]["all"]]
        xs_center = [x for x, _ in metrics[method]["center"]]
        ys_center = [y for _, y in metrics[method]["center"]]
        ax_all.plot(xs_all, ys_all, marker="o", linewidth=2, color=method_colors[method], label=method_labels[method])
        ax_center.plot(
            xs_center,
            ys_center,
            marker="s",
            linewidth=2,
            color=method_colors[method],
            label=method_labels[method],
        )

    for ax in (ax_all, ax_center):
        ax.set_xscale("log")
        ax.set_xticks(scales)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.25)
        ax.set_xlabel("Dataset size multiplier")
        ax.legend(loc="best")

    ax_all.set_title("Mean var(f) on all test x")
    ax_center.set_title("Mean var(f) on center region")
    ax_all.set_ylabel("Posterior predictive variance")
    fig2.tight_layout()
    trend_path = output_dir / "laplace_scaling_variance_vs_size.png"
    fig2.savefig(trend_path, dpi=220)
    plt.close(fig2)
    print(f"Saved: {trend_path}")

    csv_path = results_dir / "laplace_scaling_metrics.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
