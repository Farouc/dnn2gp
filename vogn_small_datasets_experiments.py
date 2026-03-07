import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neural_networks import SimpleMLP
from vogn import VOGN


@dataclass
class DatasetSpec:
    name: str
    n_train: int
    x_range: tuple[float, float]
    sigma_noise: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_gap_sine(n_train: int, x_range: tuple[float, float], sigma_noise: float):
    x = np.random.uniform(x_range[0], x_range[1], size=n_train * 3)
    mask = (x < 0.8) | (x > 2.7)
    x = x[mask][:n_train]
    x = np.sort(x)
    y_true = np.sin(1.35 * x) + 0.25 * np.cos(0.7 * x)
    y = y_true + np.random.normal(0.0, sigma_noise, size=x.shape[0])
    return x[:, None], y, y_true


def make_hetero_sine(n_train: int, x_range: tuple[float, float], sigma_noise: float):
    x = np.sort(np.random.uniform(x_range[0], x_range[1], size=n_train))
    noise_scale = sigma_noise * (0.5 + 1.8 * (1 / (1 + np.exp(-(x - 1.0)))))
    y_true = np.sin(1.2 * x)
    y = y_true + np.random.normal(0.0, noise_scale, size=x.shape[0])
    return x[:, None], y, y_true


def make_cubic(n_train: int, x_range: tuple[float, float], sigma_noise: float):
    x = np.sort(np.random.uniform(x_range[0], x_range[1], size=n_train))
    y_true = 0.08 * x**3 - 0.65 * x
    y = y_true + np.random.normal(0.0, sigma_noise, size=x.shape[0])
    return x[:, None], y, y_true


def train_vogn_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sigma_noise: float,
    device: torch.device,
    epochs: int = 1200,
    batch_size: int = 32,
    hidden_size: int = 64,
    n_layers: int = 3,
    lr: float = 0.06,
    prior_prec: float = 0.35,
    initial_prec: float = 25.0,
    train_mc_samples: int = 8,
    pred_mc_samples: int = 400,
):
    model = SimpleMLP(input_size=1, h_size=hidden_size, n_layers=n_layers, activation="tanh").to(device)
    optimizer = VOGN(
        model,
        train_set_size=x_train.shape[0],
        prior_prec=prior_prec,
        lr=lr,
        betas=(0.9, 0.995),
        num_samples=train_mc_samples,
        inital_prec=initial_prec,
    )

    xt = torch.from_numpy(x_train).double()
    yt = torch.from_numpy(y_train).double()
    x_test_t = torch.from_numpy(x_test).double().to(device)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)

    def objective(pred, target):
        return -torch.distributions.Normal(target, sigma_noise).log_prob(pred).sum()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            def closure():
                optimizer.zero_grad()
                output = model(xb).flatten()
                loss = objective(output, yb)
                return loss, output, None

            optimizer.step(closure)

        if (epoch + 1) % 300 == 0:
            with torch.no_grad():
                pred_train = model(xt.to(device)).flatten()
                train_loss = objective(pred_train, yt.to(device)).item()
            print(f"[VOGN] epoch={epoch+1} train_nll={train_loss:.4f}")

    pred_mc = torch.stack(
        optimizer.get_mc_predictions(model.forward, x_test_t, mc_samples=pred_mc_samples),
        dim=0,
    )
    mean = pred_mc.mean(0).detach().cpu().numpy().flatten()
    var = pred_mc.var(0).detach().cpu().numpy().flatten()
    return mean, var


def run_dataset_experiment(
    dataset_fn,
    spec: DatasetSpec,
    x_test: np.ndarray,
    device: torch.device,
    out_fig: Path,
    epochs: int,
    pred_mc_samples: int,
):
    x_train, y_train, y_true_train = dataset_fn(spec.n_train, spec.x_range, spec.sigma_noise)
    y_true_test = dataset_fn.__name__
    if dataset_fn is make_gap_sine:
        f = lambda x: np.sin(1.35 * x) + 0.25 * np.cos(0.7 * x)
    elif dataset_fn is make_hetero_sine:
        f = lambda x: np.sin(1.2 * x)
    else:
        f = lambda x: 0.08 * x**3 - 0.65 * x
    y_true_test = f(x_test[:, 0])

    mean, var = train_vogn_regression(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        sigma_noise=spec.sigma_noise,
        device=device,
        epochs=epochs,
        pred_mc_samples=pred_mc_samples,
    )

    std = np.sqrt(np.clip(var + spec.sigma_noise**2, 1e-12, None))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(x_train[:, 0], y_train, s=16, c="black", alpha=0.7, label="train data")
    ax.plot(x_test[:, 0], y_true_test, color="gray", linestyle="--", linewidth=1.5, label="ground truth")
    ax.plot(x_test[:, 0], mean, color="#d62728", linewidth=2.2, label="VOGN mean")
    ax.fill_between(x_test[:, 0], mean - 2 * std, mean + 2 * std, alpha=0.22, color="#d62728", label="±2 std")
    ax.set_title(f"VOGN on {spec.name} (n={spec.n_train})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    return {
        "dataset": spec.name,
        "n_train": int(spec.n_train),
        "sigma_noise": float(spec.sigma_noise),
        "mean_predictive_var": float(np.mean(var)),
        "median_predictive_var": float(np.median(var)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run VOGN experiments on multiple small synthetic datasets.")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1200)
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

    specs = [
        (make_gap_sine, DatasetSpec(name="gap_sine", n_train=120, x_range=(-3.5, 6.0), sigma_noise=0.14)),
        (make_hetero_sine, DatasetSpec(name="hetero_sine", n_train=120, x_range=(-4.0, 6.0), sigma_noise=0.10)),
        (make_cubic, DatasetSpec(name="cubic", n_train=120, x_range=(-3.0, 3.0), sigma_noise=0.24)),
    ]
    x_test = np.linspace(-4.0, 6.5, 700)[:, None]

    summary = {"device": str(device), "seed": args.seed, "datasets": []}
    for fn, spec in specs:
        print(f"\n=== Running {spec.name} ===")
        out_fig = output_dir / f"vogn_{spec.name}.png"
        metrics = run_dataset_experiment(
            dataset_fn=fn,
            spec=spec,
            x_test=x_test,
            device=device,
            out_fig=out_fig,
            epochs=args.epochs,
            pred_mc_samples=args.pred_mc_samples,
        )
        summary["datasets"].append(metrics)
        print(f"Saved: {out_fig}")

    out_json = results_dir / "vogn_small_datasets_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics: {out_json}")


if __name__ == "__main__":
    main()
