from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist_adversarial_uncertainty_experiment import laplace_gp_uncertainty_reparam_mc
from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_balanced_test_subset,
    set_seed,
)
from src.neural_networks import LeNet5


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST label-noise robustness and reliability: DNN vs GP.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--base-checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    p.add_argument("--output-dir", type=str, default="results/adversarial_examples")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--noise-levels", type=str, default="0.0,0.1,0.2,0.3,0.4")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-per-class", type=int, default=50, help="Balanced clean test subset per class.")
    p.add_argument("--prior-prec", type=float, default=1e-4)
    p.add_argument("--laplace-train-size", type=int, default=2000)
    p.add_argument("--laplace-batch-size", type=int, default=64)
    p.add_argument("--gp-mc-samples", type=int, default=50)
    p.add_argument("--figure-curves", type=str, default="mnist_label_noise_curves.png")
    p.add_argument("--figure-reliability", type=str, default="mnist_label_noise_reliability.png")
    p.add_argument("--metrics-csv", type=str, default="mnist_label_noise_metrics.csv")
    return p.parse_args()


def parse_noise_levels(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("No noise levels parsed.")
    return vals


def corrupt_labels(labels: torch.Tensor, noise: float, seed: int) -> torch.Tensor:
    if noise <= 0.0:
        return labels.clone()
    gen = torch.Generator().manual_seed(seed)
    y = labels.clone()
    n = y.numel()
    k = int(round(noise * n))
    idx = torch.randperm(n, generator=gen)[:k]
    rand = torch.randint(low=0, high=9, size=(k,), generator=gen, dtype=torch.long)
    # ensure new label differs from old by mapping in [0,8] then shifting
    y[idx] = (y[idx] + 1 + rand) % 10
    return y


def nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(-np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, None))))


def brier_multiclass(probs: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> float:
    y = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    corr = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if np.any(m):
            out += (np.sum(m) / labels.shape[0]) * abs(np.mean(conf[m]) - np.mean(corr[m]))
    return float(out)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    corr = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if np.any(m):
            bin_conf.append(float(np.mean(conf[m])))
            bin_acc.append(float(np.mean(corr[m])))
    return np.array(bin_conf), np.array(bin_acc)


def train_noisy_model(
    base_state: dict,
    x_train: torch.Tensor,
    y_noisy: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> torch.nn.Module:
    model = LeNet5(input_channels=1, dims=28, num_classes=10).to(device=device, dtype=torch.double)
    model.load_state_dict(base_state)
    model.train()

    ds = TensorDataset(x_train, y_noisy)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device=device, dtype=torch.double)
            yb = yb.to(device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def evaluate_dnn(model: torch.nn.Module, x: torch.Tensor, y: np.ndarray, device: torch.device) -> dict[str, float | np.ndarray]:
    with torch.no_grad():
        probs = torch.softmax(model(x.to(device=device, dtype=torch.double)), dim=1).detach().cpu().numpy()
    pred = np.argmax(probs, axis=1)
    return {
        "probs": probs,
        "acc": float(np.mean(pred == y)),
        "nll": nll_from_probs(probs, y),
        "brier": brier_multiclass(probs, y),
        "ece": ece(probs, y),
    }


def evaluate_gp(
    model: torch.nn.Module,
    post_prec: torch.Tensor,
    x: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> dict[str, float | np.ndarray]:
    out = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=x,
        post_prec=post_prec,
        device=device,
        mc_samples=mc_samples,
        seed=seed,
    )
    probs = out["probs"]
    pred = np.argmax(probs, axis=1)
    return {
        "probs": probs,
        "acc": float(np.mean(pred == y)),
        "nll": nll_from_probs(probs, y),
        "brier": brier_multiclass(probs, y),
        "ece": ece(probs, y),
    }


def save_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    header = [
        "noise",
        "dnn_acc",
        "gp_acc",
        "dnn_nll",
        "gp_nll",
        "dnn_brier",
        "gp_brier",
        "dnn_ece",
        "gp_ece",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(f"{r[k]:.8f}" for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(path: Path, rows: list[dict[str, float]]) -> None:
    n = np.array([r["noise"] for r in rows], dtype=float)
    dnn_acc = np.array([r["dnn_acc"] for r in rows], dtype=float)
    gp_acc = np.array([r["gp_acc"] for r in rows], dtype=float)
    dnn_nll = np.array([r["dnn_nll"] for r in rows], dtype=float)
    gp_nll = np.array([r["gp_nll"] for r in rows], dtype=float)
    dnn_ece = np.array([r["dnn_ece"] for r in rows], dtype=float)
    gp_ece = np.array([r["gp_ece"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    axes[0].plot(n, 100 * dnn_acc, "-o", color="#1F4E79", label="DNN", linewidth=2)
    axes[0].plot(n, 100 * gp_acc, "-o", color="#A61C3C", label="GP", linewidth=2)
    axes[0].set_xlabel("Train label noise fraction")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(n, dnn_nll, "-o", color="#1F4E79", label="DNN", linewidth=2)
    axes[1].plot(n, gp_nll, "-o", color="#A61C3C", label="GP", linewidth=2)
    axes[1].set_xlabel("Train label noise fraction")
    axes[1].set_ylabel("NLL")
    axes[1].grid(True)

    axes[2].plot(n, dnn_ece, "-o", color="#1F4E79", label="DNN", linewidth=2)
    axes[2].plot(n, gp_ece, "-o", color="#A61C3C", label="GP", linewidth=2)
    axes[2].set_xlabel("Train label noise fraction")
    axes[2].set_ylabel("ECE")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(path: Path, curves: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
    noise_levels = sorted(curves.keys())
    ncols = min(3, len(noise_levels))
    nrows = int(np.ceil(len(noise_levels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(noise_levels):]:
        ax.axis("off")

    for i, noise in enumerate(noise_levels):
        ax = axes[i]
        dnn_c, dnn_a = curves[noise]["dnn"]
        gp_c, gp_a = curves[noise]["gp"]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.75)
        ax.plot(dnn_c, dnn_a, "-o", color="#1F4E79", label="DNN", linewidth=1.8, markersize=4)
        ax.plot(gp_c, gp_a, "-o", color="#A61C3C", label="GP", linewidth=1.8, markersize=4)
        ax.set_title(f"Noise={noise:.2f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_fig = out_dir / args.figure_curves
    rel_fig = out_dir / args.figure_reliability
    metrics_csv = out_dir / args.metrics_csv

    noise_levels = parse_noise_levels(args.noise_levels)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Noise levels: {noise_levels}")

    base_model = load_mnist_model(Path(args.base_checkpoint), device=device)
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    train_set, test_set = load_mnist_data(args.data_dir)
    x_train = torch.as_tensor(train_set.data).unsqueeze(1).to(torch.double) / 255.0
    y_train = torch.as_tensor(train_set.targets, dtype=torch.long)
    x_test, y_test = sample_balanced_test_subset(test_set, n_per_class=args.eval_per_class, seed=args.seed + 2)
    y_test_np = y_test.numpy()

    rows: list[dict[str, float]] = []
    rel_curves: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for i, noise in enumerate(noise_levels):
        y_noisy = corrupt_labels(y_train, noise, seed=args.seed + 100 + i)
        model = train_noisy_model(
            base_state=base_state,
            x_train=x_train,
            y_noisy=y_noisy,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        dnn_metrics = evaluate_dnn(model, x_test, y_test_np, device=device)

        noisy_train_obj = SimpleNamespace(
            data=train_set.data,
            targets=y_noisy,
        )
        pp_cache = out_dir / f"mnist_laplace_post_prec_noise_{noise:.2f}.pt"
        post_prec = compute_or_load_post_prec(
            model=model,
            train_set=noisy_train_obj,
            device=device,
            cache_path=pp_cache,
            prior_prec=args.prior_prec,
            train_subset_size=args.laplace_train_size,
            batch_size=args.laplace_batch_size,
            seed=args.seed + 200 + i,
        )
        gp_metrics = evaluate_gp(
            model=model,
            post_prec=post_prec,
            x=x_test,
            y=y_test_np,
            device=device,
            mc_samples=args.gp_mc_samples,
            seed=args.seed + 300 + i,
        )

        rel_curves[noise] = {
            "dnn": reliability_curve(dnn_metrics["probs"], y_test_np),
            "gp": reliability_curve(gp_metrics["probs"], y_test_np),
        }

        row = {
            "noise": float(noise),
            "dnn_acc": float(dnn_metrics["acc"]),
            "gp_acc": float(gp_metrics["acc"]),
            "dnn_nll": float(dnn_metrics["nll"]),
            "gp_nll": float(gp_metrics["nll"]),
            "dnn_brier": float(dnn_metrics["brier"]),
            "gp_brier": float(gp_metrics["brier"]),
            "dnn_ece": float(dnn_metrics["ece"]),
            "gp_ece": float(gp_metrics["ece"]),
        }
        rows.append(row)
        print(
            f"noise={noise:.2f} | "
            f"DNN acc={100*row['dnn_acc']:.1f}% ECE={row['dnn_ece']:.4f} | "
            f"GP acc={100*row['gp_acc']:.1f}% ECE={row['gp_ece']:.4f}"
        )

    save_metrics_csv(metrics_csv, rows)
    plot_curves(curves_fig, rows)
    plot_reliability(rel_fig, rel_curves)
    print(f"Saved figure: {curves_fig}")
    print(f"Saved figure: {rel_fig}")
    print(f"Saved metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
