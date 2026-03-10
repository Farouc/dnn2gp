from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mnist_adversarial_uncertainty_experiment import (
    assign_parameters_from_vector,
    fgsm_attack,
    flatten_parameters,
    laplace_gp_uncertainty_reparam_mc,
)
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST adversarial transferability: DNN attack vs GP-surrogate attack.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    p.add_argument("--output-dir", type=str, default="results/adversarial_examples")
    p.add_argument("--figure-name", type=str, default="mnist_adversarial_transferability.png")
    p.add_argument("--matrix-name", type=str, default="mnist_adversarial_transferability_matrix_last_eps.png")
    p.add_argument("--metrics-name", type=str, default="mnist_adversarial_transferability_metrics.csv")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--epsilons", type=str, default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--prior-prec", type=float, default=1e-4)
    p.add_argument("--laplace-train-size", type=int, default=2000)
    p.add_argument("--laplace-batch-size", type=int, default=64)
    p.add_argument("--gp-mc-samples", type=int, default=80, help="MC samples for GP evaluation.")
    p.add_argument("--gp-attack-samples", type=int, default=8, help="Weight samples for GP-surrogate attack objective.")
    p.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    return p.parse_args()


def parse_eps(e: str) -> list[float]:
    vals = [float(x.strip()) for x in e.split(",") if x.strip()]
    if not vals:
        raise ValueError("No epsilon values parsed.")
    return vals


def dnn_probs(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)


def fgsm_attack_gp_surrogate(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> torch.Tensor:
    model.eval()
    theta_star = flatten_parameters(model).to(device=device, dtype=torch.double)
    precision = post_prec.to(device=device, dtype=torch.double).clamp_min(1e-12)
    std = torch.rsqrt(precision)
    generator = torch.Generator(device=device).manual_seed(seed)

    x_base = images.clone().to(device=device, dtype=torch.double)
    y = labels.to(device=device, dtype=torch.long)

    grad_accum = torch.zeros_like(x_base)
    for _ in range(mc_samples):
        x = x_base.clone().detach().requires_grad_(True)
        eps = torch.randn(theta_star.shape, generator=generator, device=device, dtype=torch.double)
        theta_sample = theta_star + std * eps
        assign_parameters_from_vector(model, theta_sample)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        grad_x = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        grad_accum = grad_accum + grad_x.detach()

    assign_parameters_from_vector(model, theta_star)
    grad_mean = grad_accum / mc_samples
    adv = torch.clamp(x_base + epsilon * grad_mean.sign(), 0.0, 1.0)
    return adv.detach().cpu()


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    header = [
        "epsilon",
        "attack_source",
        "eval_model",
        "accuracy",
        "entropy_mean",
        "one_minus_maxprob_mean",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    f"{float(r['epsilon']):.8f}",
                    str(r["attack_source"]),
                    str(r["eval_model"]),
                    f"{float(r['accuracy']):.8f}",
                    f"{float(r['entropy_mean']):.8f}",
                    f"{float(r['one_minus_maxprob_mean']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_curves_plot(path: Path, rows: list[dict[str, float | str]]) -> None:
    eps = sorted({float(r["epsilon"]) for r in rows})

    def series(attack: str, model: str, key: str) -> np.ndarray:
        vals = []
        for e in eps:
            rr = [r for r in rows if float(r["epsilon"]) == e and r["attack_source"] == attack and r["eval_model"] == model]
            vals.append(float(rr[0][key]))
        return np.array(vals, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    palette = {
        ("dnn", "dnn"): "#1F4E79",
        ("dnn", "gp"): "#3E7CB1",
        ("gp_surrogate", "dnn"): "#A61C3C",
        ("gp_surrogate", "gp"): "#D1495B",
    }

    for attack in ("dnn", "gp_surrogate"):
        for model in ("dnn", "gp"):
            label = f"attack={attack}, eval={model}"
            c = palette[(attack, model)]
            acc = series(attack, model, "accuracy")
            ent = series(attack, model, "entropy_mean")
            omm = series(attack, model, "one_minus_maxprob_mean")
            axes[0].plot(eps, 100 * acc, "-o", color=c, label=label, linewidth=2, markersize=4)
            axes[1].plot(eps, ent, "-o", color=c, label=label, linewidth=2, markersize=4)
            axes[2].plot(eps, omm, "-o", color=c, label=label, linewidth=2, markersize=4)

    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("Entropy mean")
    axes[2].set_ylabel("1 - max_prob mean")
    for ax in axes:
        ax.set_xlabel("FGSM epsilon")
        ax.grid(True)
    axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_last_eps_matrix(path: Path, rows: list[dict[str, float | str]]) -> None:
    eps_last = max(float(r["epsilon"]) for r in rows)
    rlast = [r for r in rows if float(r["epsilon"]) == eps_last]
    attacks = ["dnn", "gp_surrogate"]
    evals = ["dnn", "gp"]

    acc = np.zeros((2, 2), dtype=float)
    ent = np.zeros((2, 2), dtype=float)
    for i, a in enumerate(attacks):
        for j, m in enumerate(evals):
            rr = [r for r in rlast if r["attack_source"] == a and r["eval_model"] == m][0]
            acc[i, j] = float(rr["accuracy"])
            ent[i, j] = float(rr["entropy_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    im1 = axes[0].imshow(acc * 100, cmap="Blues", vmin=0, vmax=100)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["eval DNN", "eval GP"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["attack DNN", "attack GP"])
    axes[0].set_xlabel(f"epsilon={eps_last:.2f}")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{100*acc[i,j]:.1f}%", ha="center", va="center", fontsize=10)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(ent, cmap="Reds")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["eval DNN", "eval GP"])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["attack DNN", "attack GP"])
    axes[1].set_xlabel(f"epsilon={eps_last:.2f}")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{ent[i,j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

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
    matrix_path = out_dir / args.matrix_name
    csv_path = out_dir / args.metrics_name

    cache_path = Path(args.post_prec_cache)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path

    epsilons = parse_eps(args.epsilons)
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

    rows: list[dict[str, float | str]] = []
    for i, eps in enumerate(epsilons):
        adv_from_dnn = fgsm_attack(model, clean_images, labels, epsilon=eps, device=device)
        adv_from_gp = fgsm_attack_gp_surrogate(
            model=model,
            images=clean_images,
            labels=labels,
            epsilon=eps,
            post_prec=post_prec,
            device=device,
            mc_samples=args.gp_attack_samples,
            seed=args.seed + 50 + i,
        )

        attack_sets = {"dnn": adv_from_dnn, "gp_surrogate": adv_from_gp}
        for attack_name, adv_images in attack_sets.items():
            probs_dnn = dnn_probs(model, adv_images, device=device)
            pred_dnn = np.argmax(probs_dnn, axis=1)
            rows.append(
                {
                    "epsilon": eps,
                    "attack_source": attack_name,
                    "eval_model": "dnn",
                    "accuracy": float(np.mean(pred_dnn == labels_np)),
                    "entropy_mean": float(np.mean(entropy_from_probs(probs_dnn))),
                    "one_minus_maxprob_mean": float(np.mean(1.0 - np.max(probs_dnn, axis=1))),
                }
            )

            gp_out = laplace_gp_uncertainty_reparam_mc(
                model=model,
                images=adv_images,
                post_prec=post_prec,
                device=device,
                mc_samples=args.gp_mc_samples,
                seed=args.seed + 100 + i,
            )
            rows.append(
                {
                    "epsilon": eps,
                    "attack_source": attack_name,
                    "eval_model": "gp",
                    "accuracy": float(np.mean(gp_out["pred"] == labels_np)),
                    "entropy_mean": float(np.mean(gp_out["entropy"])),
                    "one_minus_maxprob_mean": float(np.mean(1.0 - gp_out["max_prob"])),
                }
            )

        dnn_self = [r for r in rows if r["epsilon"] == eps and r["attack_source"] == "dnn" and r["eval_model"] == "dnn"][0]
        gp_self = [r for r in rows if r["epsilon"] == eps and r["attack_source"] == "gp_surrogate" and r["eval_model"] == "gp"][0]
        print(
            f"eps={eps:.2f} | dnn-self acc={100*float(dnn_self['accuracy']):.1f}% | "
            f"gp-self acc={100*float(gp_self['accuracy']):.1f}%"
        )

    write_csv(csv_path, rows)
    make_curves_plot(fig_path, rows)
    make_last_eps_matrix(matrix_path, rows)
    print(f"Saved figure: {fig_path}")
    print(f"Saved figure: {matrix_path}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
