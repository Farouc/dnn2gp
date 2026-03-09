from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mnist_adversarial_uncertainty_experiment import fgsm_attack, laplace_gp_uncertainty_reparam_mc
from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_balanced_test_subset,
    set_seed,
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST adversarial epsilon-sweep: DNN vs reparameterized GP.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    p.add_argument("--output-dir", type=str, default="results/adversarial_examples")
    p.add_argument("--figure-name", type=str, default="mnist_adversarial_epsilon_sweep.png")
    p.add_argument("--metrics-name", type=str, default="mnist_adversarial_epsilon_sweep_metrics.csv")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--epsilons", type=str, default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--prior-prec", type=float, default=1e-4)
    p.add_argument("--laplace-train-size", type=int, default=2000)
    p.add_argument("--laplace-batch-size", type=int, default=64)
    p.add_argument("--gp-mc-samples", type=int, default=80)
    p.add_argument("--gp-mc-seed-offset", type=int, default=4)
    p.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    return p.parse_args()


def parse_epsilons(eps: str) -> list[float]:
    vals = [float(x.strip()) for x in eps.split(",") if x.strip()]
    if not vals:
        raise ValueError("No epsilons parsed.")
    return vals


def dnn_probs(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def entropy_from_probs(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    header = [
        "epsilon",
        "dnn_acc",
        "gp_acc",
        "dnn_entropy_mean",
        "gp_entropy_mean",
        "dnn_1_minus_maxprob_mean",
        "gp_1_minus_maxprob_mean",
        "gp_epistemic_sum_mean",
        "gp_aleatoric_sum_mean",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(f"{r[k]:.8f}" for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, rows: list[dict[str, float]]) -> None:
    eps = np.array([r["epsilon"] for r in rows], dtype=float)
    dnn_acc = np.array([r["dnn_acc"] for r in rows], dtype=float)
    gp_acc = np.array([r["gp_acc"] for r in rows], dtype=float)
    dnn_ent = np.array([r["dnn_entropy_mean"] for r in rows], dtype=float)
    gp_ent = np.array([r["gp_entropy_mean"] for r in rows], dtype=float)
    dnn_1m = np.array([r["dnn_1_minus_maxprob_mean"] for r in rows], dtype=float)
    gp_1m = np.array([r["gp_1_minus_maxprob_mean"] for r in rows], dtype=float)
    gp_epi = np.array([r["gp_epistemic_sum_mean"] for r in rows], dtype=float)
    gp_alea = np.array([r["gp_aleatoric_sum_mean"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(eps, dnn_acc * 100, "-o", color="#1F4E79", label="DNN")
    ax1.plot(eps, gp_acc * 100, "-o", color="#A61C3C", label="GP (reparam)")
    ax1.set_xlabel("FGSM epsilon")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.plot(eps, dnn_ent, "-o", color="#1F4E79", label="DNN")
    ax2.plot(eps, gp_ent, "-o", color="#A61C3C", label="GP (reparam)")
    ax2.set_xlabel("FGSM epsilon")
    ax2.set_ylabel("Mean entropy")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3.plot(eps, dnn_1m, "-o", color="#1F4E79", label="DNN")
    ax3.plot(eps, gp_1m, "-o", color="#A61C3C", label="GP (reparam)")
    ax3.set_xlabel("FGSM epsilon")
    ax3.set_ylabel("Mean 1 - max_prob")
    ax3.grid(True)
    ax3.legend(loc="best")

    ax4.plot(eps, gp_epi, "-o", color="#B8860B", label="GP epistemic sum")
    ax4.plot(eps, gp_alea, "-o", color="#5F9EA0", label="GP aleatoric sum")
    ax4.set_xlabel("FGSM epsilon")
    ax4.set_ylabel("Mean GP uncertainty sum")
    ax4.grid(True)
    ax4.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / args.figure_name
    csv_path = out_dir / args.metrics_name

    cache_path = Path(args.post_prec_cache)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path

    epsilons = parse_epsilons(args.epsilons)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Epsilons: {epsilons}")

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
    labels_np = labels.numpy()

    rows: list[dict[str, float]] = []
    for i, eps in enumerate(epsilons):
        adv_images = fgsm_attack(model, clean_images, labels, epsilon=eps, device=device)

        dnn_p = dnn_probs(model, adv_images, device=device)
        dnn_pred = np.argmax(dnn_p, axis=1)
        dnn_acc = float(np.mean(dnn_pred == labels_np))
        dnn_ent = entropy_from_probs(dnn_p)
        dnn_1m = 1.0 - np.max(dnn_p, axis=1)

        gp_out = laplace_gp_uncertainty_reparam_mc(
            model=model,
            images=adv_images,
            post_prec=post_prec,
            device=device,
            mc_samples=args.gp_mc_samples,
            seed=args.seed + args.gp_mc_seed_offset + i,
        )
        gp_acc = float(np.mean(gp_out["pred"] == labels_np))
        gp_ent = gp_out["entropy"]
        gp_1m = 1.0 - gp_out["max_prob"]
        gp_epi = gp_out["epistemic"].sum(axis=1)
        gp_alea = gp_out["aleatoric"].sum(axis=1)

        row = {
            "epsilon": float(eps),
            "dnn_acc": dnn_acc,
            "gp_acc": gp_acc,
            "dnn_entropy_mean": float(np.mean(dnn_ent)),
            "gp_entropy_mean": float(np.mean(gp_ent)),
            "dnn_1_minus_maxprob_mean": float(np.mean(dnn_1m)),
            "gp_1_minus_maxprob_mean": float(np.mean(gp_1m)),
            "gp_epistemic_sum_mean": float(np.mean(gp_epi)),
            "gp_aleatoric_sum_mean": float(np.mean(gp_alea)),
        }
        rows.append(row)
        print(
            f"eps={eps:.2f} | dnn_acc={100*dnn_acc:.1f}% gp_acc={100*gp_acc:.1f}% | "
            f"dnn_ent={row['dnn_entropy_mean']:.4f} gp_ent={row['gp_entropy_mean']:.4f}"
        )

    write_csv(csv_path, rows)
    make_plot(fig_path, rows)
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()

