from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    image_strip,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_balanced_test_subset,
    set_seed,
)


def set_neurips_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "legend.edgecolor": "#BBBBBB",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
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


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def assign_parameters_from_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[idx : idx + n].view_as(p))
            idx += n


def laplace_gp_predictive_probs_mc(
    model: torch.nn.Module,
    images: torch.Tensor,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    theta_map = flatten_parameters(model).to(device=device, dtype=torch.double)
    precision = post_prec.to(device=device, dtype=torch.double).clamp_min(1e-12)
    std = torch.rsqrt(precision)
    x = images.to(device=device, dtype=torch.double)
    generator = torch.Generator(device=device).manual_seed(seed)

    probs_sum = None
    with torch.no_grad():
        for _ in range(mc_samples):
            eps = torch.randn(theta_map.shape, generator=generator, device=device, dtype=torch.double)
            theta_sample = theta_map + std * eps
            assign_parameters_from_vector(model, theta_sample)
            probs = torch.softmax(model(x), dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

    assign_parameters_from_vector(model, theta_map)
    return (probs_sum / mc_samples).detach().cpu().numpy()


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads, dim=0)


def laplace_gp_uncertainty_reparam_mc(
    model: torch.nn.Module,
    images: torch.Tensor,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    model.eval()
    theta_star = flatten_parameters(model).to(device=device, dtype=torch.double)
    inv_post_prec = torch.reciprocal(post_prec.to(device=device, dtype=torch.double).clamp_min(1e-12))
    generator = torch.Generator(device=device).manual_seed(seed)
    probs_mean_list = []
    entropy_list = []
    max_prob_list = []
    pred_list = []
    epistemic_list = []
    aleatoric_list = []

    for i in range(images.shape[0]):
        x = images[i : i + 1].to(device=device, dtype=torch.double)
        model.zero_grad(set_to_none=True)
        logits = model(x)[0]  # [C]
        p = torch.softmax(logits, dim=0)
        lam = torch.diag(p) - torch.outer(p, p)

        jac_rows = []
        for cls in range(logits.shape[0]):
            logits[cls].backward(retain_graph=(cls < logits.shape[0] - 1))
            jac_rows.append(_gradient_vector(model))
            model.zero_grad(set_to_none=True)
        jac = torch.stack(jac_rows, dim=0)  # [C,P]
        jtheta = jac @ theta_star  # [C]

        cov_f = torch.einsum("cp,p,dp->cd", jac, inv_post_prec, jac)
        cov_f = 0.5 * (cov_f + cov_f.T)
        eye = torch.eye(cov_f.shape[0], device=device, dtype=torch.double)
        lam_reg = lam + 1e-6 * eye
        lam_inv = torch.linalg.pinv(lam_reg)
        cov_tilde = cov_f + lam_inv
        cov_tilde = 0.5 * (cov_tilde + cov_tilde.T) + 1e-6 * eye
        chol = torch.linalg.cholesky(cov_tilde)

        eps = torch.randn((mc_samples, logits.shape[0]), generator=generator, device=device, dtype=torch.double)
        y_tilde_samples = jtheta.unsqueeze(0) + eps @ chol.T  # [S,C]
        delta = y_tilde_samples - jtheta.unsqueeze(0)  # [S,C]
        # Multiclass local map inversion analogue of Eq. 46: y = p + Lambda (y_tilde - J theta*)
        probs_samples = p.unsqueeze(0) + delta @ lam.T
        probs_samples = torch.clamp(probs_samples, min=1e-9)
        probs_samples = probs_samples / probs_samples.sum(dim=1, keepdim=True)

        probs_mean = probs_samples.mean(dim=0)  # [C]

        epistemic = probs_samples.var(dim=0, unbiased=False)  # [C]
        aleatoric = (probs_samples * (1.0 - probs_samples)).mean(dim=0)  # [C]
        entropy = -torch.sum(probs_mean * torch.log(probs_mean.clamp_min(1e-12)))
        max_prob = probs_mean.max()
        pred = torch.argmax(probs_mean)

        probs_mean_list.append(probs_mean)
        entropy_list.append(entropy)
        max_prob_list.append(max_prob)
        pred_list.append(pred)
        epistemic_list.append(epistemic)
        aleatoric_list.append(aleatoric)

    return {
        "probs": torch.stack(probs_mean_list, dim=0).detach().cpu().numpy(),
        "epistemic": torch.stack(epistemic_list, dim=0).detach().cpu().numpy(),
        "aleatoric": torch.stack(aleatoric_list, dim=0).detach().cpu().numpy(),
        "entropy": torch.stack(entropy_list, dim=0).detach().cpu().numpy(),
        "max_prob": torch.stack(max_prob_list, dim=0).detach().cpu().numpy(),
        "pred": torch.stack(pred_list, dim=0).detach().cpu().numpy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST adversarial uncertainty with Laplace dnn2gp.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--figures-subdir", type=str, default="adversarial_examples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--n-per-class", type=int, default=3, help="Balanced test subset per digit.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="FGSM perturbation size.")
    parser.add_argument("--prior-prec", type=float, default=1e-4)
    parser.add_argument("--laplace-train-size", type=int, default=2000)
    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--gp-mc-samples", type=int, default=80)
    parser.add_argument("--gp-mc-seed-offset", type=int, default=4)
    parser.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    parser.add_argument(
        "--shared-adv-cache",
        type=str,
        default="results/mnist_adversarial_examples_shared.npz",
    )
    parser.add_argument("--figure-name", type=str, default="mnist_adversarial_uncertainty.png")
    parser.add_argument("--gp-vs-dnn-figure-name", type=str, default="mnist_adversarial_gp_vs_dnn_confidence.png")
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


def plot_samples_strip_figure(path: Path, clean_images: torch.Tensor, adv_images: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    clean_strip = image_strip(clean_images, max_items=12)
    adv_strip = image_strip(adv_images, max_items=12)
    sep = np.zeros((4, clean_strip.shape[1]), dtype=np.float64)
    stacked = np.concatenate([clean_strip, sep, adv_strip], axis=0)
    ax.imshow(stacked, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_shift_figure(
    path: Path,
    clean: dict[str, np.ndarray],
    adv: dict[str, np.ndarray],
) -> None:
    epi_clean = clean["epistemic"].sum(axis=1)
    epi_adv = adv["epistemic"].sum(axis=1)
    alea_clean = clean["aleatoric"].sum(axis=1)
    alea_adv = adv["aleatoric"].sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    ax.scatter(np.clip(alea_clean, 1e-8, None), np.clip(epi_clean, 1e-8, None), s=36, alpha=0.85, label="clean", c="#1f77b4")
    ax.scatter(np.clip(alea_adv, 1e-8, None), np.clip(epi_adv, 1e-8, None), s=36, alpha=0.85, label="adversarial", c="#d62728")
    ax.set_xlabel("Aleatoric uncertainty")
    ax.set_ylabel("Epistemic uncertainty")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gp_averages_figure(
    path: Path,
    labels: np.ndarray,
    clean: dict[str, np.ndarray],
    adv: dict[str, np.ndarray],
) -> None:
    epi_clean = clean["epistemic"].sum(axis=1)
    epi_adv = adv["epistemic"].sum(axis=1)
    alea_clean = clean["aleatoric"].sum(axis=1)
    alea_adv = adv["aleatoric"].sum(axis=1)

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
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.bar(x - width / 2, clean_means, width=width, color="#1F4E79", label="clean")
    ax.bar(x + width / 2, adv_means, width=width, color="#A61C3C", label="adversarial")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=14)
    ax.set_ylabel("Average value")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")
    for i in range(len(metric_names)):
        ax.text(x[i] - width / 2, clean_means[i], f"{clean_means[i]:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, adv_means[i], f"{adv_means[i]:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dnn_predictive_metrics(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    entropy = (-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).detach().cpu().numpy()
    one_minus_max_prob = (1.0 - probs.max(dim=1).values).detach().cpu().numpy()
    return entropy, one_minus_max_prob


def plot_gp_vs_dnn_adversarial_confidence(
    path: Path,
    gp_adv: dict[str, np.ndarray],
    dnn_adv_entropy: np.ndarray,
    dnn_adv_one_minus_max_prob: np.ndarray,
) -> None:
    gp_adv_entropy = gp_adv["entropy"]
    gp_adv_one_minus_max_prob = 1.0 - gp_adv["max_prob"]

    labels = [
        "Entropy (DNN)",
        "Entropy (GP)",
        "1-MaxProb (DNN)",
        "1-MaxProb (GP)",
    ]
    values = [
        float(np.mean(dnn_adv_entropy)),
        float(np.mean(gp_adv_entropy)),
        float(np.mean(dnn_adv_one_minus_max_prob)),
        float(np.mean(gp_adv_one_minus_max_prob)),
    ]
    dnn_color = "#1F4E79"
    gp_color = "#A61C3C"
    colors = [dnn_color, gp_color, dnn_color, gp_color]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.64)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14)
    ax.set_ylabel("Average value on adversarial examples")
    ax.grid(axis="y", alpha=0.25)

    for xi, val in zip(x, values):
        ax.text(xi, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_neurips_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / args.figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / args.figure_name
    gp_vs_dnn_fig_path = figures_dir / args.gp_vs_dnn_figure_name
    samples_fig_path = figures_dir / "mnist_adversarial_samples_clean_vs_adv.png"
    shift_fig_path = figures_dir / "mnist_adversarial_uncertainty_shift.png"
    gp_avg_fig_path = figures_dir / "mnist_adversarial_gp_averages_clean_vs_adv.png"
    metrics_path = output_dir / args.metrics_name
    shared_adv_cache_path = Path(args.shared_adv_cache)
    if not shared_adv_cache_path.is_absolute():
        shared_adv_cache_path = Path.cwd() / shared_adv_cache_path
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

    if shared_adv_cache_path.exists():
        shared = np.load(shared_adv_cache_path)
        clean_images = torch.from_numpy(shared["clean_images"]).to(torch.double)
        adv_images = torch.from_numpy(shared["adv_images"]).to(torch.double)
        labels = torch.from_numpy(shared["labels"]).to(torch.long)
        print(f"Loaded shared adversarial cache: {shared_adv_cache_path}")
    else:
        clean_images, labels = sample_balanced_test_subset(test_set, n_per_class=args.n_per_class, seed=args.seed + 2)
        adv_images = fgsm_attack(model, clean_images, labels, epsilon=args.epsilon, device=device)
        shared_adv_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shared_adv_cache_path,
            clean_images=clean_images.numpy(),
            adv_images=adv_images.numpy(),
            labels=labels.numpy(),
        )
        print(f"Saved shared adversarial cache: {shared_adv_cache_path}")

    print("Computing Laplace GP predictive uncertainties (clean)...")
    clean_out = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=clean_images,
        post_prec=post_prec,
        device=device,
        mc_samples=args.gp_mc_samples,
        seed=args.seed + args.gp_mc_seed_offset - 1,
    )
    print("Computing Laplace GP predictive uncertainties (adversarial)...")
    adv_out = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=adv_images,
        post_prec=post_prec,
        device=device,
        mc_samples=args.gp_mc_samples,
        seed=args.seed + args.gp_mc_seed_offset,
    )

    if np.allclose(clean_out["entropy"], adv_out["entropy"]) and np.allclose(clean_out["max_prob"], adv_out["max_prob"]):
        raise RuntimeError(
            "Clean and adversarial GP entropy/max-prob are numerically identical; check GP probability pipeline."
        )

    print(
        "GP means | "
        f"entropy_clean={np.mean(clean_out['entropy']):.6f}, entropy_adv={np.mean(adv_out['entropy']):.6f}, "
        f"one_minus_max_clean={np.mean(1.0 - clean_out['max_prob']):.6f}, "
        f"one_minus_max_adv={np.mean(1.0 - adv_out['max_prob']):.6f}"
    )
    dnn_adv_entropy, dnn_adv_one_minus_max_prob = dnn_predictive_metrics(model, adv_images, device=device)

    labels_np = labels.numpy()
    save_metrics_csv(metrics_path, labels_np, clean_out, adv_out)
    plot_samples_strip_figure(samples_fig_path, clean_images, adv_images)
    plot_uncertainty_shift_figure(shift_fig_path, clean_out, adv_out)
    plot_gp_averages_figure(gp_avg_fig_path, labels_np, clean_out, adv_out)
    plot_gp_vs_dnn_adversarial_confidence(
        gp_vs_dnn_fig_path,
        adv_out,
        dnn_adv_entropy,
        dnn_adv_one_minus_max_prob,
    )
    print(f"Saved figure: {gp_vs_dnn_fig_path}")
    print(f"Saved figure: {samples_fig_path}")
    print(f"Saved figure: {shift_fig_path}")
    print(f"Saved figure: {gp_avg_fig_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
