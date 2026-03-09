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
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_balanced_test_subset,
    set_seed,
)
from mnist_adversarial_uncertainty_experiment import laplace_gp_uncertainty_reparam_mc


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


def dnn_predictive_probs(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


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


def laplace_gp_predictive_probs_reparam_mc(
    model: torch.nn.Module,
    images: torch.Tensor,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    theta_star = flatten_parameters(model).to(device=device, dtype=torch.double)
    inv_post_prec = torch.reciprocal(post_prec.to(device=device, dtype=torch.double).clamp_min(1e-12))
    generator = torch.Generator(device=device).manual_seed(seed)
    probs_mean_list = []

    for i in range(images.shape[0]):
        x = images[i : i + 1].to(device=device, dtype=torch.double)
        model.zero_grad(set_to_none=True)
        logits = model(x)[0]
        p = torch.softmax(logits, dim=0)
        lam = torch.diag(p) - torch.outer(p, p)

        jac_rows = []
        for cls in range(logits.shape[0]):
            logits[cls].backward(retain_graph=(cls < logits.shape[0] - 1))
            jac_rows.append(_gradient_vector(model))
            model.zero_grad(set_to_none=True)
        jac = torch.stack(jac_rows, dim=0)
        jtheta = jac @ theta_star

        cov_f = torch.einsum("cp,p,dp->cd", jac, inv_post_prec, jac)
        cov_f = 0.5 * (cov_f + cov_f.T)
        eye = torch.eye(cov_f.shape[0], device=device, dtype=torch.double)
        lam_reg = lam + 1e-6 * eye
        lam_inv = torch.linalg.pinv(lam_reg)
        cov_tilde = cov_f + lam_inv
        cov_tilde = 0.5 * (cov_tilde + cov_tilde.T) + 1e-6 * eye
        chol = torch.linalg.cholesky(cov_tilde)

        eps = torch.randn((mc_samples, logits.shape[0]), generator=generator, device=device, dtype=torch.double)
        y_tilde_samples = jtheta.unsqueeze(0) + eps @ chol.T
        delta = y_tilde_samples - jtheta.unsqueeze(0)
        probs_samples = p.unsqueeze(0) + delta @ lam.T
        probs_samples = torch.clamp(probs_samples, min=1e-9)
        probs_samples = probs_samples / probs_samples.sum(dim=1, keepdim=True)
        probs_mean_list.append(probs_samples.mean(dim=0))

    return torch.stack(probs_mean_list, dim=0).detach().cpu().numpy()


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)


def one_minus_max_prob(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probs, axis=1)


def plot_gp_vs_dnn_adv_confidence(
    path: Path,
    dnn_probs_adv: np.ndarray,
    gp_probs_adv: np.ndarray,
) -> dict[str, float]:
    dnn_entropy = entropy_from_probs(dnn_probs_adv)
    gp_entropy = entropy_from_probs(gp_probs_adv)
    dnn_ommp = one_minus_max_prob(dnn_probs_adv)
    gp_ommp = one_minus_max_prob(gp_probs_adv)

    labels = ["Entropy (DNN)", "Entropy (GP)", "1-MaxProb (DNN)", "1-MaxProb (GP)"]
    means = [
        float(np.mean(dnn_entropy)),
        float(np.mean(gp_entropy)),
        float(np.mean(dnn_ommp)),
        float(np.mean(gp_ommp)),
    ]
    colors = ["#ff7f0e", "#2ca02c", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.64, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_ylabel("Average value on adversarial examples")
    ax.set_title("MNIST adversarial confidence: DNN vs Laplace GP predictive probs")
    ax.grid(axis="y", alpha=0.25)

    for xi, val in zip(x, means):
        ax.text(xi, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "dnn_entropy_mean": means[0],
        "gp_entropy_mean": means[1],
        "dnn_1_minus_maxprob_mean": means[2],
        "gp_1_minus_maxprob_mean": means[3],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DNN vs Laplace GP confidence on MNIST adversarial examples."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--figure-name",
        type=str,
        default="mnist_adversarial_gp_vs_dnn_confidence_proper_probs.png",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / args.figure_name

    cache_path = Path(args.post_prec_cache)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    shared_adv_cache_path = Path(args.shared_adv_cache)
    if not shared_adv_cache_path.is_absolute():
        shared_adv_cache_path = Path.cwd() / shared_adv_cache_path

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

    print("Computing DNN predictive probabilities on adversarial examples...")
    dnn_probs_adv = dnn_predictive_probs(model, adv_images, device=device)

    print("Computing Laplace GP predictive probabilities (reparameterized) on adversarial examples...")
    gp_out_adv = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=adv_images,
        post_prec=post_prec,
        device=device,
        mc_samples=args.gp_mc_samples,
        seed=args.seed + args.gp_mc_seed_offset,
    )
    gp_probs_adv = gp_out_adv["probs"]

    metrics = plot_gp_vs_dnn_adv_confidence(
        path=figure_path,
        dnn_probs_adv=dnn_probs_adv,
        gp_probs_adv=gp_probs_adv,
    )

    print(f"Saved figure: {figure_path}")
    print(
        "Means | "
        f"DNN entropy={metrics['dnn_entropy_mean']:.6f}, "
        f"GP entropy={metrics['gp_entropy_mean']:.6f}, "
        f"DNN 1-max={metrics['dnn_1_minus_maxprob_mean']:.6f}, "
        f"GP 1-max={metrics['gp_1_minus_maxprob_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
