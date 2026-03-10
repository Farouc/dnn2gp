from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_regression_mean_variance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    title: str,
    path: Path,
) -> None:
    std = np.sqrt(np.clip(var, 1e-12, None))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.scatter(x_train, y_train, s=12, c="black", alpha=0.7, label="train")
    ax.plot(x_test, mean, color="#d62728", linewidth=2.2, label="predictive mean")
    ax.fill_between(x_test, mean - 2.0 * std, mean + 2.0 * std, color="#d62728", alpha=0.2, label="±2 std")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, path)


def plot_gp_vs_mc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    gp_mean: np.ndarray,
    gp_var: np.ndarray,
    mc_mean: np.ndarray,
    mc_var: np.ndarray,
    title: str,
    path: Path,
) -> None:
    gp_std = np.sqrt(np.clip(gp_var, 1e-12, None))
    mc_std = np.sqrt(np.clip(mc_var, 1e-12, None))
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.5), sharex=True)

    axs[0].scatter(x_train, y_train, s=10, c="black", alpha=0.65, label="train")
    axs[0].plot(x_test, gp_mean, color="#d62728", linewidth=2.0, label="GP mean")
    axs[0].plot(x_test, mc_mean, color="#1f77b4", linewidth=2.0, linestyle="--", label="MC mean")
    axs[0].fill_between(x_test, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color="#d62728", alpha=0.16)
    axs[0].fill_between(x_test, mc_mean - 2 * mc_std, mc_mean + 2 * mc_std, color="#1f77b4", alpha=0.16)
    axs[0].set_title("Predictive mean")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="best")

    axs[1].plot(x_test, gp_var, color="#d62728", linewidth=2.0, label="GP variance")
    axs[1].plot(x_test, mc_var, color="#1f77b4", linewidth=2.0, linestyle="--", label="MC variance")
    axs[1].set_title("Predictive variance")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("var")
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="best")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, path)


def plot_kernel_heatmap(
    kernel: np.ndarray,
    title: str,
    path: Path,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(kernel, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_fig(fig, path)


def plot_metric_curves(
    x_values: np.ndarray,
    series: dict[str, np.ndarray],
    title: str,
    y_label: str,
    path: Path,
    x_label: str = "Iteration / Epoch",
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    marker = None if len(x_values) > 25 else "o"
    for idx, (name, y) in enumerate(series.items()):
        ax.plot(x_values, y, marker=marker, linewidth=2.0, color=palette[idx % len(palette)], label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, path)


def plot_uncertainty_histograms(
    seen_scores: np.ndarray,
    unseen_scores: np.ndarray,
    title: str,
    x_label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    ax.hist(seen_scores, bins=35, alpha=0.6, color="#1f77b4", density=True, label="seen (0/1)")
    ax.hist(unseen_scores, bins=35, alpha=0.6, color="#d62728", density=True, label="unseen (2-9)")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, path)


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    n_bins: int = 15,
) -> None:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if np.any(mask):
            centers.append((lo + hi) * 0.5)
            accs.append(np.mean(correct[mask]))
        else:
            centers.append((lo + hi) * 0.5)
            accs.append(np.nan)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.3, label="perfect calibration")
    ax.plot(centers, accs, marker="o", linewidth=2.0, color="#1f77b4", label="model")
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, path)
