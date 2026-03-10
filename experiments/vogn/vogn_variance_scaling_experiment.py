import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neural_networks import SimpleMLP
from vogn import VOGN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sine_dataset(n: int, sigma_noise: float, x_low: float = -4.0, x_high: float = 4.0):
    x = np.sort(np.random.uniform(x_low, x_high, size=n))
    y_true = np.sin(1.4 * x) + 0.15 * np.cos(0.7 * x)
    y = y_true + np.random.normal(0.0, sigma_noise, size=n)
    return x[:, None], y


def train_vogn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sigma_noise: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    pred_mc_samples: int,
):
    model = SimpleMLP(input_size=1, h_size=64, n_layers=3, activation="tanh").to(device)
    optimizer = VOGN(
        model,
        train_set_size=x_train.shape[0],
        prior_prec=0.4,
        lr=0.05,
        betas=(0.9, 0.995),
        num_samples=6,
        inital_prec=30.0,
    )

    xt = torch.from_numpy(x_train).double()
    yt = torch.from_numpy(y_train).double()
    test_t = torch.from_numpy(x_test).double().to(device)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)

    def objective(pred, target):
        return -torch.distributions.Normal(target, sigma_noise).log_prob(pred).sum()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            def closure():
                optimizer.zero_grad()
                out = model(xb).flatten()
                loss = objective(out, yb)
                return loss, out, None

            optimizer.step(closure)

        if (epoch + 1) % max(100, epochs // 5) == 0:
            with torch.no_grad():
                pred = model(xt.to(device)).flatten()
                nll = objective(pred, yt.to(device)).item()
            print(f"  epoch={epoch+1}/{epochs} train_nll={nll:.4f}")

    pred_mc = torch.stack(
        optimizer.get_mc_predictions(model.forward, test_t, mc_samples=pred_mc_samples),
        dim=0,
    )
    mean = pred_mc.mean(0).detach().cpu().numpy().flatten()
    var = pred_mc.var(0).detach().cpu().numpy().flatten()
    return mean, var


def parse_args():
    parser = argparse.ArgumentParser(
        description="VOGN scaling experiment: dataset sizes x10/x100/x1000 and posterior variance trend."
    )
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--base-n", type=int, default=10, help="Base dataset size; scales produce N,10N,100N,1000N.")
    parser.add_argument("--sigma-noise", type=float, default=0.20)
    parser.add_argument("--pred-mc-samples", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.double)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scales = [1, 10, 100, 1000]
    ns = [args.base_n * s for s in scales]
    epochs_by_scale = {1: 2500, 10: 1400, 100: 700, 1000: 220}
    batch_by_scale = {1: 32, 10: 64, 100: 128, 1000: 256}

    x_test = np.linspace(-4.2, 4.2, 600)[:, None]
    y_true_test = np.sin(1.4 * x_test[:, 0]) + 0.15 * np.cos(0.7 * x_test[:, 0])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    rows = ["scale,n_train,epochs,batch_size,mean_var_all,mean_var_center,std_var_all"]

    for idx, (scale, n_train) in enumerate(zip(scales, ns)):
        print(f"\n=== Scale x{scale} (n_train={n_train}) ===")
        x_train, y_train = make_sine_dataset(n=n_train, sigma_noise=args.sigma_noise)

        mean, var = train_vogn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            sigma_noise=args.sigma_noise,
            device=device,
            epochs=epochs_by_scale[scale],
            batch_size=batch_by_scale[scale],
            pred_mc_samples=args.pred_mc_samples,
        )

        center_mask = (x_test[:, 0] >= -2.0) & (x_test[:, 0] <= 2.0)
        mean_var_all = float(np.mean(var))
        mean_var_center = float(np.mean(var[center_mask]))
        std_var_all = float(np.std(var))
        rows.append(
            f"{scale},{n_train},{epochs_by_scale[scale]},{batch_by_scale[scale]},"
            f"{mean_var_all:.8f},{mean_var_center:.8f},{std_var_all:.8f}"
        )

        ax = axs[idx]
        std = np.sqrt(np.clip(var + args.sigma_noise**2, 1e-12, None))
        stride = max(1, n_train // 500)
        ax.scatter(x_train[::stride, 0], y_train[::stride], s=8, c="black", alpha=0.5, label="train")
        ax.plot(x_test[:, 0], y_true_test, color="gray", linestyle="--", linewidth=1.3, label="ground truth")
        ax.plot(x_test[:, 0], mean, color="#1f77b4", linewidth=2.0, label="VOGN mean")
        ax.fill_between(x_test[:, 0], mean - 2 * std, mean + 2 * std, alpha=0.2, color="#1f77b4", label="±2 std")
        ax.set_title(f"x{scale} (N={n_train})")
        ax.grid(alpha=0.25)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    pred_path = output_dir / "vogn_scaling_predictive.png"
    fig.savefig(pred_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pred_path}")

    data = np.genfromtxt(rows[1:], delimiter=",")
    scales_arr = data[:, 0]
    mean_var_all = data[:, 4]
    mean_var_center = data[:, 5]

    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6))
    ax2.plot(scales_arr, mean_var_all, marker="o", linewidth=2, color="#d62728", label="mean var (all test x)")
    ax2.plot(scales_arr, mean_var_center, marker="s", linewidth=2, color="#2ca02c", label="mean var (center x)")
    ax2.set_xscale("log")
    ax2.set_xticks(scales)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel("Dataset size multiplier")
    ax2.set_ylabel("Posterior predictive variance")
    ax2.set_title("VOGN variance vs dataset size")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    fig2.tight_layout()
    trend_path = output_dir / "vogn_scaling_variance_vs_size.png"
    fig2.savefig(trend_path, dpi=220)
    plt.close(fig2)
    print(f"Saved: {trend_path}")

    csv_path = results_dir / "vogn_scaling_metrics.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
