from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST transformation uncertainty with Laplace dnn2gp.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--n-per-class", type=int, default=1, help="Base images sampled per digit.")
    parser.add_argument("--rotation-min", type=float, default=-60.0)
    parser.add_argument("--rotation-max", type=float, default=60.0)
    parser.add_argument("--rotation-steps", type=int, default=13)
    parser.add_argument("--translation-max", type=int, default=8, help="Max horizontal shift in pixels.")
    parser.add_argument("--translation-step", type=int, default=2)
    parser.add_argument("--prior-prec", type=float, default=1e-4)
    parser.add_argument("--laplace-train-size", type=int, default=2000)
    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    parser.add_argument("--figure-name", type=str, default="mnist_transformation_uncertainty.png")
    parser.add_argument("--metrics-name", type=str, default="mnist_transformation_uncertainty_metrics.csv")
    return parser.parse_args()


def build_rotation_grid(images: torch.Tensor, angles: np.ndarray):
    transformed = []
    meta = []
    for a in angles:
        for i in range(images.shape[0]):
            img = TF.rotate(
                images[i],
                angle=float(a),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            transformed.append(img)
            meta.append((i, float(a)))
    return torch.stack(transformed, dim=0), meta


def build_translation_grid(images: torch.Tensor, shifts: np.ndarray):
    transformed = []
    meta = []
    for s in shifts:
        for i in range(images.shape[0]):
            img = TF.affine(
                images[i],
                angle=0.0,
                translate=[int(s), 0],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            transformed.append(img)
            meta.append((i, int(s)))
    return torch.stack(transformed, dim=0), meta


def summarize_by_param(
    out: dict[str, np.ndarray],
    params: np.ndarray,
    n_base: int,
) -> dict[str, np.ndarray]:
    epi = out["epistemic"].sum(axis=1).reshape(len(params), n_base)
    alea = out["aleatoric"].sum(axis=1).reshape(len(params), n_base)
    entropy = out["entropy"].reshape(len(params), n_base)
    max_prob = out["max_prob"].reshape(len(params), n_base)
    return {
        "epi_mean": epi.mean(axis=1),
        "epi_std": epi.std(axis=1),
        "alea_mean": alea.mean(axis=1),
        "alea_std": alea.std(axis=1),
        "entropy_mean": entropy.mean(axis=1),
        "entropy_std": entropy.std(axis=1),
        "maxprob_mean": max_prob.mean(axis=1),
        "maxprob_std": max_prob.std(axis=1),
    }


def save_metrics_csv(
    path: Path,
    labels: np.ndarray,
    rotation_angles: np.ndarray,
    rotation_out: dict[str, np.ndarray],
    translation_shifts: np.ndarray,
    translation_out: dict[str, np.ndarray],
    n_base: int,
) -> None:
    lines = [
        "transform_type,param_value,sample_id,true_label,pred,entropy,epi_sum,alea_sum,max_prob",
    ]
    rot_epi = rotation_out["epistemic"].sum(axis=1).reshape(len(rotation_angles), n_base)
    rot_alea = rotation_out["aleatoric"].sum(axis=1).reshape(len(rotation_angles), n_base)
    rot_entropy = rotation_out["entropy"].reshape(len(rotation_angles), n_base)
    rot_pred = rotation_out["pred"].reshape(len(rotation_angles), n_base)
    rot_maxp = rotation_out["max_prob"].reshape(len(rotation_angles), n_base)
    for i, angle in enumerate(rotation_angles):
        for j in range(n_base):
            lines.append(
                f"rotation,{angle:.4f},{j},{int(labels[j])},{int(rot_pred[i,j])},"
                f"{rot_entropy[i,j]:.8f},{rot_epi[i,j]:.8f},{rot_alea[i,j]:.8f},{rot_maxp[i,j]:.8f}"
            )

    tr_epi = translation_out["epistemic"].sum(axis=1).reshape(len(translation_shifts), n_base)
    tr_alea = translation_out["aleatoric"].sum(axis=1).reshape(len(translation_shifts), n_base)
    tr_entropy = translation_out["entropy"].reshape(len(translation_shifts), n_base)
    tr_pred = translation_out["pred"].reshape(len(translation_shifts), n_base)
    tr_maxp = translation_out["max_prob"].reshape(len(translation_shifts), n_base)
    for i, shift in enumerate(translation_shifts):
        for j in range(n_base):
            lines.append(
                f"translation,{int(shift)},{j},{int(labels[j])},{int(tr_pred[i,j])},"
                f"{tr_entropy[i,j]:.8f},{tr_epi[i,j]:.8f},{tr_alea[i,j]:.8f},{tr_maxp[i,j]:.8f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_transform_figure(
    path: Path,
    rep_rot_images: torch.Tensor,
    rep_tr_images: torch.Tensor,
    rotation_angles: np.ndarray,
    translation_shifts: np.ndarray,
    rot_stats: dict[str, np.ndarray],
    tr_stats: dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.26)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_strip(rep_rot_images, max_items=rep_rot_images.shape[0]), cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Representative digit under rotations")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(rotation_angles, np.clip(rot_stats["epi_mean"], 1e-10, None), color="#d62728", linewidth=2.2, label="epistemic")
    ax1.plot(rotation_angles, np.clip(rot_stats["alea_mean"], 1e-10, None), color="#1f77b4", linewidth=2.2, label="aleatoric")
    ax1.set_xlabel("Rotation angle (degrees)")
    ax1.set_ylabel("Uncertainty (sum over classes)")
    ax1.set_title("Rotation vs uncertainty")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    ax1b = ax1.twinx()
    ax1b.plot(rotation_angles, rot_stats["maxprob_mean"], color="#2ca02c", linestyle="--", linewidth=1.8, label="max prob")
    ax1b.set_ylabel("Max class probability")
    ax1b.set_ylim(0.0, 1.02)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(image_strip(rep_tr_images, max_items=rep_tr_images.shape[0]), cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title("Representative digit under horizontal translations")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(translation_shifts, np.clip(tr_stats["epi_mean"], 1e-10, None), color="#d62728", linewidth=2.2, label="epistemic")
    ax3.plot(translation_shifts, np.clip(tr_stats["alea_mean"], 1e-10, None), color="#1f77b4", linewidth=2.2, label="aleatoric")
    ax3.set_xlabel("Horizontal translation (pixels)")
    ax3.set_ylabel("Uncertainty (sum over classes)")
    ax3.set_title("Translation vs uncertainty")
    ax3.set_yscale("log")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")
    ax3b = ax3.twinx()
    ax3b.plot(
        translation_shifts,
        tr_stats["maxprob_mean"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="max prob",
    )
    ax3b.set_ylabel("Max class probability")
    ax3b.set_ylim(0.0, 1.02)

    fig.suptitle("MNIST Laplace dnn2gp: uncertainty under geometric transformations", fontsize=14)
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

    base_images, base_labels = sample_balanced_test_subset(test_set, n_per_class=args.n_per_class, seed=args.seed + 2)
    n_base = base_images.shape[0]
    rotation_angles = np.linspace(args.rotation_min, args.rotation_max, args.rotation_steps)
    translation_shifts = np.arange(-args.translation_max, args.translation_max + 1, args.translation_step)

    rot_images, _ = build_rotation_grid(base_images, rotation_angles)
    tr_images, _ = build_translation_grid(base_images, translation_shifts)

    print("Computing uncertainty for rotated images...")
    rot_out = compute_uncertainty_from_post_prec(model, rot_images, post_prec, device=device)
    print("Computing uncertainty for translated images...")
    tr_out = compute_uncertainty_from_post_prec(model, tr_images, post_prec, device=device)

    rot_stats = summarize_by_param(rot_out, rotation_angles, n_base=n_base)
    tr_stats = summarize_by_param(tr_out, translation_shifts, n_base=n_base)

    rep_idx = int(torch.where(base_labels == 3)[0][0].item()) if (base_labels == 3).any() else 0
    rot_rep = []
    rep_angles = np.linspace(args.rotation_min, args.rotation_max, 9)
    for a in rep_angles:
        rot_rep.append(
            TF.rotate(
                base_images[rep_idx],
                angle=float(a),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        )
    tr_rep = []
    rep_shifts = np.arange(-args.translation_max, args.translation_max + 1, max(1, args.translation_step * 2))
    for s in rep_shifts:
        tr_rep.append(
            TF.affine(
                base_images[rep_idx],
                angle=0.0,
                translate=[int(s), 0],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        )

    save_metrics_csv(
        metrics_path,
        labels=base_labels.numpy(),
        rotation_angles=rotation_angles,
        rotation_out=rot_out,
        translation_shifts=translation_shifts,
        translation_out=tr_out,
        n_base=n_base,
    )
    plot_transform_figure(
        fig_path,
        rep_rot_images=torch.stack(rot_rep, dim=0),
        rep_tr_images=torch.stack(tr_rep, dim=0),
        rotation_angles=rotation_angles,
        translation_shifts=translation_shifts,
        rot_stats=rot_stats,
        tr_stats=tr_stats,
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
