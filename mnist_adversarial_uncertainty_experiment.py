from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    compute_uncertainty_from_post_prec,
    image_strip,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_balanced_test_subset,
    set_seed,
)


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    x = images.clone().to(device=device, dtype=torch.double).requires_grad_(True)
    y = labels.to(device=device, dtype=torch.long)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adv = torch.clamp(x + epsilon * x.grad.sign(), 0.0, 1.0)
    return adv.detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST adversarial uncertainty with Laplace dnn2gp.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--n-per-class", type=int, default=3, help="Balanced test subset per digit.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="FGSM perturbation size.")
    parser.add_argument("--prior-prec", type=float, default=1e-4)
    parser.add_argument("--laplace-train-size", type=int, default=2000)
    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    parser.add_argument("--figure-name", type=str, default="mnist_adversarial_uncertainty.png")
    parser.add_argument("--metrics-name", type=str, default="mnist_adversarial_uncertainty_metrics.csv")
    return parser.parse_args()


def save_metrics_csv(
    path: Path,
    labels: np.ndarray,
    clean: dict[str, np.ndarray],
    adv: dict[str, np.ndarray],
) -> None:
    header = (
        "idx,true_label,pred_clean,pred_adv,attack_success,"
        "entropy_clean,entropy_adv,epi_clean,epi_adv,alea_clean,alea_adv,maxprob_clean,maxprob_adv"
    )
    epi_clean = clean["epistemic"].sum(axis=1)
    epi_adv = adv["epistemic"].sum(axis=1)
    alea_clean = clean["aleatoric"].sum(axis=1)
    alea_adv = adv["aleatoric"].sum(axis=1)
    attack_success = (adv["pred"] != labels).astype(int)
    rows = []
    for i in range(labels.shape[0]):
        rows.append(
            ",".join(
                [
                    str(i),
                    str(int(labels[i])),
                    str(int(clean["pred"][i])),
                    str(int(adv["pred"][i])),
                    str(int(attack_success[i])),
                    f"{clean['entropy'][i]:.8f}",
                    f"{adv['entropy'][i]:.8f}",
                    f"{epi_clean[i]:.8f}",
                    f"{epi_adv[i]:.8f}",
                    f"{alea_clean[i]:.8f}",
                    f"{alea_adv[i]:.8f}",
                    f"{clean['max_prob'][i]:.8f}",
                    f"{adv['max_prob'][i]:.8f}",
                ]
            )
        )
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def plot_adversarial_figure(
    path: Path,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: np.ndarray,
    clean: dict[str, np.ndarray],
    adv: dict[str, np.ndarray],
) -> None:
    epi_clean = clean["epistemic"].sum(axis=1)
    epi_adv = adv["epistemic"].sum(axis=1)
    alea_clean = clean["aleatoric"].sum(axis=1)
    alea_adv = adv["aleatoric"].sum(axis=1)

    clean_acc = float(np.mean(clean["pred"] == labels))
    adv_acc = float(np.mean(adv["pred"] == labels))
    attack_rate = float(np.mean(adv["pred"] != labels))

    fig = plt.figure(figsize=(13.0, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.27, wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_strip(clean_images, max_items=12), cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Clean MNIST samples")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(image_strip(adv_images, max_items=12), cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("FGSM adversarial samples")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(np.clip(alea_clean, 1e-8, None), np.clip(epi_clean, 1e-8, None), s=36, alpha=0.8, label="clean", c="#1f77b4")
    ax2.scatter(np.clip(alea_adv, 1e-8, None), np.clip(epi_adv, 1e-8, None), s=36, alpha=0.8, label="adversarial", c="#d62728")
    ax2.set_xlabel("Aleatoric uncertainty (sum over classes)")
    ax2.set_ylabel("Epistemic uncertainty (sum over classes)")
    ax2.set_title("Uncertainty shift under attack")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(gs[1, 1])
    metric_names = ["Entropy", "Epistemic sum", "Aleatoric sum", "1 - Max prob"]
    clean_means = [
        np.mean(clean["entropy"]),
        np.mean(epi_clean),
        np.mean(alea_clean),
        np.mean(1.0 - clean["max_prob"]),
    ]
    adv_means = [
        np.mean(adv["entropy"]),
        np.mean(epi_adv),
        np.mean(alea_adv),
        np.mean(1.0 - adv["max_prob"]),
    ]
    x = np.arange(len(metric_names))
    width = 0.36
    ax3.bar(x - width / 2, clean_means, width=width, color="#1f77b4", label="clean")
    ax3.bar(x + width / 2, adv_means, width=width, color="#d62728", label="adversarial")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, rotation=15)
    ax3.set_title(
        "Metric averages\n"
        f"clean_acc={100*clean_acc:.1f}% | adv_acc={100*adv_acc:.1f}% | attack_success={100*attack_rate:.1f}%"
    )
    ax3.set_yscale("log")
    ax3.grid(alpha=0.25, axis="y")
    ax3.legend(loc="best")

    fig.suptitle("MNIST Laplace dnn2gp: adversarial uncertainty analysis", fontsize=14)
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

    clean_images, labels = sample_balanced_test_subset(test_set, n_per_class=args.n_per_class, seed=args.seed + 2)
    adv_images = fgsm_attack(model, clean_images, labels, epsilon=args.epsilon, device=device)

    print("Computing clean uncertainties...")
    clean_out = compute_uncertainty_from_post_prec(model, clean_images, post_prec, device=device)
    print("Computing adversarial uncertainties...")
    adv_out = compute_uncertainty_from_post_prec(model, adv_images, post_prec, device=device)

    labels_np = labels.numpy()
    save_metrics_csv(metrics_path, labels_np, clean_out, adv_out)
    plot_adversarial_figure(fig_path, clean_images, adv_images, labels_np, clean_out, adv_out)
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
