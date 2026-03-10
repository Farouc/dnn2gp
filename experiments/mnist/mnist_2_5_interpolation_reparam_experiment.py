from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mnist_adversarial_uncertainty_experiment import laplace_gp_uncertainty_reparam_mc
from mnist_dnn2gp_experiment_utils import (
    compute_or_load_post_prec,
    image_strip,
    load_mnist_data,
    load_mnist_model,
    resolve_device,
    sample_digit_pairs,
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
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST 2->5 interpolation with reparameterized GP vs DNN.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    p.add_argument("--output-dir", type=str, default="results/adversarial_examples")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--digit-a", type=int, default=2)
    p.add_argument("--digit-b", type=int, default=5)
    p.add_argument("--n-pairs", type=int, default=1)
    p.add_argument("--n-alpha", type=int, default=41)
    p.add_argument("--prior-prec", type=float, default=1e-4)
    p.add_argument("--laplace-train-size", type=int, default=2000)
    p.add_argument("--laplace-batch-size", type=int, default=64)
    p.add_argument("--gp-mc-samples", type=int, default=80)
    p.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    p.add_argument("--figure-name", type=str, default="mnist_2_5_interpolation_reparam.png")
    p.add_argument("--fig-visualization", type=str, default="mnist_2_5_interpolation_visualization.png")
    p.add_argument("--fig-pred-class", type=str, default="mnist_2_5_predicted_class_curve.png")
    p.add_argument("--fig-gp-unc", type=str, default="mnist_2_5_gp_uncertainty_curve.png")
    p.add_argument("--fig-entropy", type=str, default="mnist_2_5_entropy_curve.png")
    p.add_argument("--fig-prob-crossover", type=str, default="mnist_2_5_class_probability_curve.png")
    p.add_argument("--metrics-name", type=str, default="mnist_2_5_interpolation_reparam_metrics.csv")
    return p.parse_args()


def interpolate_pairs(img_a: torch.Tensor, img_b: torch.Tensor, alphas: np.ndarray) -> torch.Tensor:
    items = []
    for alpha in alphas:
        a = float(alpha)
        items.append((1.0 - a) * img_a + a * img_b)
    return torch.stack(items, dim=0)


def dnn_outputs(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    p = probs.detach().cpu().numpy()
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return {
        "probs": p,
        "pred": np.argmax(p, axis=1),
        "entropy": ent,
    }


def save_metrics_csv(
    path: Path,
    alphas: np.ndarray,
    dnn: dict[str, np.ndarray],
    gp: dict[str, np.ndarray],
    n_pairs: int,
    digit_a: int,
    digit_b: int,
) -> None:
    header = (
        "pair_id,alpha,dnn_pred,gp_pred,dnn_prob_a,dnn_prob_b,gp_prob_a,gp_prob_b,"
        "dnn_entropy,gp_entropy,gp_epi_sum,gp_alea_sum"
    )
    lines = [header]
    dnn_pred = dnn["pred"].reshape(n_pairs, len(alphas))
    gp_pred = gp["pred"].reshape(n_pairs, len(alphas))
    dnn_prob_a = dnn["probs"][:, digit_a].reshape(n_pairs, len(alphas))
    dnn_prob_b = dnn["probs"][:, digit_b].reshape(n_pairs, len(alphas))
    gp_prob_a = gp["probs"][:, digit_a].reshape(n_pairs, len(alphas))
    gp_prob_b = gp["probs"][:, digit_b].reshape(n_pairs, len(alphas))
    dnn_ent = dnn["entropy"].reshape(n_pairs, len(alphas))
    gp_ent = gp["entropy"].reshape(n_pairs, len(alphas))
    gp_epi = gp["epistemic"].sum(axis=1).reshape(n_pairs, len(alphas))
    gp_alea = gp["aleatoric"].sum(axis=1).reshape(n_pairs, len(alphas))

    for i in range(n_pairs):
        for j, a in enumerate(alphas):
            lines.append(
                ",".join(
                    [
                        str(i),
                        f"{a:.6f}",
                        str(int(dnn_pred[i, j])),
                        str(int(gp_pred[i, j])),
                        f"{dnn_prob_a[i, j]:.8f}",
                        f"{dnn_prob_b[i, j]:.8f}",
                        f"{gp_prob_a[i, j]:.8f}",
                        f"{gp_prob_b[i, j]:.8f}",
                        f"{dnn_ent[i, j]:.8f}",
                        f"{gp_ent[i, j]:.8f}",
                        f"{gp_epi[i, j]:.8f}",
                        f"{gp_alea[i, j]:.8f}",
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / args.figure_name
    fig_vis_path = out_dir / args.fig_visualization
    fig_pred_path = out_dir / args.fig_pred_class
    fig_gp_unc_path = out_dir / args.fig_gp_unc
    fig_ent_path = out_dir / args.fig_entropy
    fig_prob_path = out_dir / args.fig_prob_crossover
    metrics_path = out_dir / args.metrics_name
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

    img_a, img_b = sample_digit_pairs(
        test_set=test_set,
        digit_a=args.digit_a,
        digit_b=args.digit_b,
        n_pairs=args.n_pairs,
        seed=args.seed + 2,
    )

    alphas = np.linspace(0.0, 1.0, args.n_alpha)
    trajs = []
    for i in range(args.n_pairs):
        trajs.append(interpolate_pairs(img_a[i], img_b[i], alphas))
    x_traj = torch.cat(trajs, dim=0)  # [pairs * n_alpha, 1, 28, 28]

    dnn = dnn_outputs(model, x_traj, device=device)
    gp = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=x_traj,
        post_prec=post_prec,
        device=device,
        mc_samples=args.gp_mc_samples,
        seed=args.seed + 3,
    )

    nA = len(alphas)
    nP = args.n_pairs
    dnn_pred = dnn["pred"].reshape(nP, nA)
    gp_pred = gp["pred"].reshape(nP, nA)
    dnn_ent = dnn["entropy"].reshape(nP, nA)
    gp_ent = gp["entropy"].reshape(nP, nA)
    gp_epi = gp["epistemic"].sum(axis=1).reshape(nP, nA)
    gp_alea = gp["aleatoric"].sum(axis=1).reshape(nP, nA)
    dnn_prob_a = dnn["probs"][:, args.digit_a].reshape(nP, nA)
    dnn_prob_b = dnn["probs"][:, args.digit_b].reshape(nP, nA)
    gp_prob_a = gp["probs"][:, args.digit_a].reshape(nP, nA)
    gp_prob_b = gp["probs"][:, args.digit_b].reshape(nP, nA)

    # Mean trajectories (useful for n_pairs > 1)
    dnn_pred_m = dnn_pred.mean(axis=0)
    gp_pred_m = gp_pred.mean(axis=0)
    dnn_ent_m = dnn_ent.mean(axis=0)
    gp_ent_m = gp_ent.mean(axis=0)
    gp_epi_m = gp_epi.mean(axis=0)
    gp_alea_m = gp_alea.mean(axis=0)
    dnn_pa_m = dnn_prob_a.mean(axis=0)
    dnn_pb_m = dnn_prob_b.mean(axis=0)
    gp_pa_m = gp_prob_a.mean(axis=0)
    gp_pb_m = gp_prob_b.mean(axis=0)

    # Optional extra: transition alpha where predicted class first leaves digit_a.
    def first_leave_alpha(pred_curve: np.ndarray, cls: int) -> float:
        idx = np.where(np.round(pred_curve).astype(int) != cls)[0]
        return float(alphas[idx[0]]) if idx.size > 0 else np.nan

    dnn_leave = first_leave_alpha(dnn_pred_m, args.digit_a)
    gp_leave = first_leave_alpha(gp_pred_m, args.digit_a)
    print(
        f"Transition alpha (leave class {args.digit_a}): "
        f"DNN={dnn_leave if not np.isnan(dnn_leave) else 'nan'}, "
        f"GP={gp_leave if not np.isnan(gp_leave) else 'nan'}"
    )

    # 1) Visualization figure
    fig_vis, ax_vis = plt.subplots(figsize=(8.2, 1.65))
    n_frames = trajs[0].shape[0]
    idx = np.linspace(0, n_frames - 1, num=min(12, n_frames)).astype(int)
    strip = image_strip(trajs[0][idx], max_items=len(idx), pad=2)
    ax_vis.imshow(strip, cmap="gray", vmin=0.0, vmax=1.0)
    ax_vis.axis("off")
    fig_vis.tight_layout(pad=0.02)
    fig_vis.savefig(fig_vis_path, dpi=220, bbox_inches="tight")
    plt.close(fig_vis)

    # 2) Predicted class evolution with dots
    fig_pred, ax_pred = plt.subplots(figsize=(6.6, 2.8))
    ax_pred.plot(alphas, dnn_pred_m, color="#1F4E79", linewidth=2.2, marker="o", markersize=3.5, label="DNN predicted class")
    ax_pred.plot(alphas, gp_pred_m, color="#A61C3C", linewidth=2.2, marker="o", markersize=3.5, label="GP predicted class")
    ax_pred.set_xlabel("Interpolation alpha (0: digit 2, 1: digit 5)")
    ax_pred.set_ylabel("Predicted class")
    ax_pred.set_yticks(range(10))
    ax_pred.grid(True)
    ax_pred.legend(loc="best")
    fig_pred.tight_layout(pad=0.35)
    fig_pred.savefig(fig_pred_path, dpi=220, bbox_inches="tight")
    plt.close(fig_pred)

    # 3) GP epistemic/aleatoric curve
    fig_unc, ax_unc = plt.subplots(figsize=(6.6, 2.8))
    ax_unc.plot(alphas, gp_epi_m, color="#B8860B", linewidth=2.2, marker="o", markersize=3.2, label="GP epistemic")
    ax_unc.plot(alphas, gp_alea_m, color="#2E8B57", linewidth=2.2, marker="o", markersize=3.2, label="GP aleatoric")
    ax_unc.set_xlabel("Interpolation alpha")
    ax_unc.set_ylabel("Uncertainty")
    ax_unc.grid(True)
    ax_unc.legend(loc="best")
    fig_unc.tight_layout(pad=0.35)
    fig_unc.savefig(fig_gp_unc_path, dpi=220, bbox_inches="tight")
    plt.close(fig_unc)

    # 4) Entropy evolution DNN vs GP
    fig_ent, ax_ent = plt.subplots(figsize=(6.6, 2.8))
    ax_ent.plot(alphas, dnn_ent_m, color="#1F4E79", linewidth=2.2, marker="o", markersize=3.2, label="DNN entropy")
    ax_ent.plot(alphas, gp_ent_m, color="#A61C3C", linewidth=2.2, marker="o", markersize=3.2, label="GP entropy")
    ax_ent.set_xlabel("Interpolation alpha")
    ax_ent.set_ylabel("Predictive entropy")
    ax_ent.grid(True)
    ax_ent.legend(loc="best")
    fig_ent.tight_layout(pad=0.35)
    fig_ent.savefig(fig_ent_path, dpi=220, bbox_inches="tight")
    plt.close(fig_ent)

    # 5) Additional informative figure: class probability crossover
    fig_prob, ax_prob = plt.subplots(figsize=(6.6, 2.8))
    ax_prob.plot(alphas, dnn_pa_m, "--", color="#1F4E79", linewidth=2.0, marker="o", markersize=2.8, label=f"DNN p(class {args.digit_a})")
    ax_prob.plot(alphas, dnn_pb_m, "-", color="#1F4E79", linewidth=2.0, marker="o", markersize=2.8, label=f"DNN p(class {args.digit_b})")
    ax_prob.plot(alphas, gp_pa_m, "--", color="#A61C3C", linewidth=2.0, marker="o", markersize=2.8, label=f"GP p(class {args.digit_a})")
    ax_prob.plot(alphas, gp_pb_m, "-", color="#A61C3C", linewidth=2.0, marker="o", markersize=2.8, label=f"GP p(class {args.digit_b})")
    ax_prob.set_xlabel("Interpolation alpha")
    ax_prob.set_ylabel("Class probability")
    ax_prob.set_ylim(0.0, 1.02)
    ax_prob.grid(True)
    ax_prob.legend(loc="best", fontsize=8)
    fig_prob.tight_layout(pad=0.35)
    fig_prob.savefig(fig_prob_path, dpi=220, bbox_inches="tight")
    plt.close(fig_prob)

    # Keep backward-compatible combined figure path by saving probability-crossover there.
    # This preserves previous CLI expectations while producing the requested separate PNGs.
    fig_legacy, ax_legacy = plt.subplots(figsize=(6.6, 2.8))
    ax_legacy.plot(alphas, dnn_pa_m, "--", color="#1F4E79", linewidth=2.0)
    ax_legacy.plot(alphas, dnn_pb_m, "-", color="#1F4E79", linewidth=2.0)
    ax_legacy.plot(alphas, gp_pa_m, "--", color="#A61C3C", linewidth=2.0)
    ax_legacy.plot(alphas, gp_pb_m, "-", color="#A61C3C", linewidth=2.0)
    ax_legacy.set_xlabel("Interpolation alpha")
    ax_legacy.set_ylabel("Class probability")
    ax_legacy.set_ylim(0.0, 1.02)
    ax_legacy.grid(True)
    fig_legacy.tight_layout(pad=0.35)
    fig_legacy.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig_legacy)

    save_metrics_csv(
        metrics_path,
        alphas=alphas,
        dnn=dnn,
        gp=gp,
        n_pairs=args.n_pairs,
        digit_a=args.digit_a,
        digit_b=args.digit_b,
    )

    print(f"Saved figure: {fig_vis_path}")
    print(f"Saved figure: {fig_pred_path}")
    print(f"Saved figure: {fig_gp_unc_path}")
    print(f"Saved figure: {fig_ent_path}")
    print(f"Saved figure: {fig_prob_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
