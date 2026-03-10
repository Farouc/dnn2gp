from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    compute_uncertainty_from_post_prec,
    image_strip,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_digit_pairs,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST 3<->8 interpolation uncertainty with Laplace dnn2gp.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--n-pairs", type=int, default=6)
    parser.add_argument("--alpha-steps", type=int, default=21)
    parser.add_argument("--prior-prec", type=float, default=1e-4)
    parser.add_argument("--laplace-train-size", type=int, default=2000)
    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    parser.add_argument("--figure-name", type=str, default="mnist_3_8_interpolation_uncertainty.png")
    parser.add_argument("--metrics-name", type=str, default="mnist_3_8_interpolation_uncertainty_metrics.csv")
    return parser.parse_args()


def build_interpolations(
    imgs3: torch.Tensor,
    imgs8: torch.Tensor,
    alphas: np.ndarray,
) -> torch.Tensor:
    out = []
    for p in range(imgs3.shape[0]):
        for a in alphas:
            mix = (1.0 - float(a)) * imgs3[p] + float(a) * imgs8[p]
            out.append(mix)
    return torch.stack(out, dim=0)


def save_metrics_csv(
    path: Path,
    alphas: np.ndarray,
    out: dict[str, np.ndarray],
    n_pairs: int,
) -> None:
    n_alpha = len(alphas)
    probs = out["probs"].reshape(n_pairs, n_alpha, -1)
    pred = out["pred"].reshape(n_pairs, n_alpha)
    entropy = out["entropy"].reshape(n_pairs, n_alpha)
    epi = out["epistemic"].sum(axis=1).reshape(n_pairs, n_alpha)
    alea = out["aleatoric"].sum(axis=1).reshape(n_pairs, n_alpha)
    maxp = out["max_prob"].reshape(n_pairs, n_alpha)

    lines = [
        "pair_id,alpha,pred,prob_3,prob_8,entropy,epi_sum,alea_sum,max_prob",
    ]
    for i in range(n_pairs):
        for j, alpha in enumerate(alphas):
            lines.append(
                f"{i},{alpha:.6f},{int(pred[i,j])},{probs[i,j,3]:.8f},{probs[i,j,8]:.8f},"
                f"{entropy[i,j]:.8f},{epi[i,j]:.8f},{alea[i,j]:.8f},{maxp[i,j]:.8f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_interpolation_figure(
    path: Path,
    imgs3: torch.Tensor,
    imgs8: torch.Tensor,
    alphas: np.ndarray,
    out: dict[str, np.ndarray],
    n_pairs: int,
) -> None:
    n_alpha = len(alphas)
    probs = out["probs"].reshape(n_pairs, n_alpha, -1)
    pred = out["pred"].reshape(n_pairs, n_alpha)
    entropy = out["entropy"].reshape(n_pairs, n_alpha)
    epi = out["epistemic"].sum(axis=1).reshape(n_pairs, n_alpha)
    alea = out["aleatoric"].sum(axis=1).reshape(n_pairs, n_alpha)

    pair0_mix = []
    for alpha in np.linspace(0.0, 1.0, 11):
        pair0_mix.append((1.0 - alpha) * imgs3[0] + alpha * imgs8[0])
    pair0_mix = torch.stack(pair0_mix, dim=0)

    mean_entropy = entropy.mean(axis=0)
    std_entropy = entropy.std(axis=0)
    mean_epi = epi.mean(axis=0)
    std_epi = epi.std(axis=0)
    mean_alea = alea.mean(axis=0)
    std_alea = alea.std(axis=0)

    fig = plt.figure(figsize=(13.4, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.26)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_strip(pair0_mix, max_items=pair0_mix.shape[0]), cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Pair 0 interpolation: 3 -> 8")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(alphas, probs[0, :, 3], linewidth=2.2, color="#1f77b4", label="P(class=3)")
    ax1.plot(alphas, probs[0, :, 8], linewidth=2.2, color="#d62728", label="P(class=8)")
    ax1.set_xlabel("Interpolation coefficient alpha")
    ax1.set_ylabel("Class probability")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_title("Class probabilities along interpolation (pair 0)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(alphas, np.clip(mean_epi, 1e-10, None), color="#d62728", linewidth=2.2, label="epistemic")
    ax2.plot(alphas, np.clip(mean_alea, 1e-10, None), color="#1f77b4", linewidth=2.2, label="aleatoric")
    ax2.plot(alphas, np.clip(mean_entropy, 1e-10, None), color="#2ca02c", linewidth=2.0, linestyle="--", label="entropy")
    ax2.set_xlabel("Interpolation coefficient alpha")
    ax2.set_ylabel("Uncertainty metric value")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    ax2.set_title("Uncertainty evolution (mean +/- std across pairs)")

    ax3 = fig.add_subplot(gs[1, 1])
    cmap = ListedColormap(plt.cm.tab10.colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), cmap.N)
    im = ax3.imshow(pred, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax3.set_xlabel("alpha index")
    ax3.set_ylabel("pair index")
    ax3.set_title("Predicted class along interpolation")
    ticks = np.linspace(0, n_alpha - 1, min(6, n_alpha)).astype(int)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([f"{alphas[t]:.2f}" for t in ticks])
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted class")
    cbar.set_ticks(np.arange(10))

    fig.suptitle("MNIST Laplace dnn2gp: 3<->8 interpolation uncertainty", fontsize=14)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / args.figure_name
    metrics_path = output_dir / args.metrics_name
    cache_path = Path(args.post_prec_cache)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = load_mnist_model(Path(args.checkpoint), device=device)
    train_set, test_set = load_mnist_data(args.data_dir)

    post_prec = compute_or_load_post_prec(
        model=model,
        train_set=train_set,
        device=device,
        cache_path=cache_path,
        prior_prec=args.prior_prec,
        train_subset_size=args.laplace_train_size,
        batch_size=args.laplace_batch_size,
        seed=args.seed + 1,
    )

    imgs3, imgs8 = sample_digit_pairs(test_set, digit_a=3, digit_b=8, n_pairs=args.n_pairs, seed=args.seed + 2)
    alphas = np.linspace(0.0, 1.0, args.alpha_steps)
    interp_images = build_interpolations(imgs3, imgs8, alphas)

    print("Computing uncertainty on 3<->8 interpolation images...")
    out = compute_uncertainty_from_post_prec(model, interp_images, post_prec, device=device)

    save_metrics_csv(metrics_path, alphas, out, n_pairs=args.n_pairs)
    plot_interpolation_figure(fig_path, imgs3, imgs8, alphas, out, n_pairs=args.n_pairs)
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
