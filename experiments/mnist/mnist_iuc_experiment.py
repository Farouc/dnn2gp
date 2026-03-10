from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from mnist_adversarial_uncertainty_experiment import laplace_gp_uncertainty_reparam_mc
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
    p = argparse.ArgumentParser(description="Invariance-Uncertainty Consistency (IUC) on MNIST: DNN vs GP.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    p.add_argument("--output-dir", type=str, default="results/adversarial_examples")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--n-per-class", type=int, default=1, help="Base clean images per class.")
    p.add_argument("--steps", type=int, default=11, help="Number of points per transform trajectory.")
    p.add_argument("--rotation-max-deg", type=float, default=30.0)
    p.add_argument("--translation-max-px", type=int, default=4)
    p.add_argument("--scale-min", type=float, default=0.75)
    p.add_argument("--unc-rise-fraction", type=float, default=0.25, help="Threshold fraction of trajectory uncertainty rise.")
    p.add_argument("--prior-prec", type=float, default=1e-4)
    p.add_argument("--laplace-train-size", type=int, default=2000)
    p.add_argument("--laplace-batch-size", type=int, default=64)
    p.add_argument("--gp-mc-samples", type=int, default=40)
    p.add_argument("--post-prec-cache", type=str, default="results/mnist_laplace_post_prec_2000.pt")
    p.add_argument("--traj-figure", type=str, default="mnist_iuc_trajectories.png")
    p.add_argument("--gap-figure", type=str, default="mnist_iuc_early_warning_gap.png")
    p.add_argument("--summary-csv", type=str, default="mnist_iuc_summary.csv")
    p.add_argument("--events-csv", type=str, default="mnist_iuc_events.csv")
    return p.parse_args()


def dnn_outputs(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device=device, dtype=torch.double))
        probs = torch.softmax(logits, dim=1)
    p = probs.detach().cpu().numpy()
    return {
        "probs": p,
        "pred": np.argmax(p, axis=1),
        "entropy": -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1),
        "one_minus_max": 1.0 - np.max(p, axis=1),
    }


def build_rotation_traj(images: torch.Tensor, ts: np.ndarray, max_deg: float) -> torch.Tensor:
    out = []
    for t in ts:
        angle = float(max_deg * t)
        for i in range(images.shape[0]):
            out.append(
                TF.rotate(
                    images[i],
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            )
    return torch.stack(out, dim=0)


def build_translation_traj(images: torch.Tensor, ts: np.ndarray, max_px: int) -> torch.Tensor:
    out = []
    for t in ts:
        shift = int(round(max_px * t))
        for i in range(images.shape[0]):
            out.append(
                TF.affine(
                    images[i],
                    angle=0.0,
                    translate=[shift, 0],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            )
    return torch.stack(out, dim=0)


def build_scale_traj(images: torch.Tensor, ts: np.ndarray, scale_min: float) -> torch.Tensor:
    out = []
    for t in ts:
        scale = float(1.0 - (1.0 - scale_min) * t)
        for i in range(images.shape[0]):
            out.append(
                TF.affine(
                    images[i],
                    angle=0.0,
                    translate=[0, 0],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            )
    return torch.stack(out, dim=0)


def reshape_traj(v: np.ndarray, n_steps: int, n_samples: int) -> np.ndarray:
    return v.reshape(n_steps, n_samples).T  # [N,T]


def first_flip_t(pred_row: np.ndarray, t_grid: np.ndarray) -> float:
    base = pred_row[0]
    idx = np.where(pred_row != base)[0]
    return float(t_grid[idx[0]]) if idx.size > 0 else np.nan


def first_unc_t(u_row: np.ndarray, t_grid: np.ndarray, rise_frac: float) -> float:
    u0 = float(u_row[0])
    umax = float(np.max(u_row))
    thresh = u0 + rise_frac * max(0.0, umax - u0)
    idx = np.where(u_row >= thresh)[0]
    return float(t_grid[idx[0]]) if idx.size > 0 else np.nan


def monotonicity_score(u_row: np.ndarray) -> float:
    if u_row.size < 2:
        return 1.0
    d = np.diff(u_row)
    return float(np.mean(d >= -1e-10))


def auc_uncertainty(u_row: np.ndarray, t_grid: np.ndarray) -> float:
    return float(np.trapz(u_row, x=t_grid) / (t_grid[-1] - t_grid[0] + 1e-12))


def evaluate_transform(
    name: str,
    traj_images: torch.Tensor,
    n_steps: int,
    n_samples: int,
    t_grid: np.ndarray,
    model: torch.nn.Module,
    post_prec: torch.Tensor,
    device: torch.device,
    gp_mc_samples: int,
    seed: int,
    rise_frac: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]]]:
    dnn = dnn_outputs(model, traj_images, device=device)
    gp = laplace_gp_uncertainty_reparam_mc(
        model=model,
        images=traj_images,
        post_prec=post_prec,
        device=device,
        mc_samples=gp_mc_samples,
        seed=seed,
    )

    dnn_pred = reshape_traj(dnn["pred"], n_steps, n_samples)
    dnn_ent = reshape_traj(dnn["entropy"], n_steps, n_samples)
    dnn_1m = reshape_traj(dnn["one_minus_max"], n_steps, n_samples)
    gp_pred = reshape_traj(gp["pred"], n_steps, n_samples)
    gp_ent = reshape_traj(gp["entropy"], n_steps, n_samples)
    gp_1m = reshape_traj(1.0 - gp["max_prob"], n_steps, n_samples)

    events: list[dict[str, float | str]] = []
    for i in range(n_samples):
        for model_name, pred_row, unc_row in [
            ("dnn", dnn_pred[i], dnn_ent[i]),
            ("gp", gp_pred[i], gp_ent[i]),
        ]:
            t_flip = first_flip_t(pred_row, t_grid)
            t_unc = first_unc_t(unc_row, t_grid, rise_frac)
            gap = np.nan if (np.isnan(t_flip) or np.isnan(t_unc)) else float(t_flip - t_unc)
            events.append(
                {
                    "transform": name,
                    "sample_id": float(i),
                    "model": model_name,
                    "t_flip": float(t_flip) if not np.isnan(t_flip) else np.nan,
                    "t_unc": float(t_unc) if not np.isnan(t_unc) else np.nan,
                    "gap": gap,
                    "monotonicity": monotonicity_score(unc_row),
                    "auc_uncertainty": auc_uncertainty(unc_row, t_grid),
                }
            )

    packed = {
        "t_grid": t_grid,
        "dnn_entropy_mean": dnn_ent.mean(axis=0),
        "gp_entropy_mean": gp_ent.mean(axis=0),
        "dnn_one_minus_max_mean": dnn_1m.mean(axis=0),
        "gp_one_minus_max_mean": gp_1m.mean(axis=0),
        "dnn_accuracy_vs_t": np.mean(dnn_pred == dnn_pred[:, [0]], axis=0),
        "gp_accuracy_vs_t": np.mean(gp_pred == gp_pred[:, [0]], axis=0),
    }
    return packed, events


def save_events_csv(path: Path, events: list[dict[str, float | str]]) -> None:
    header = ["transform", "sample_id", "model", "t_flip", "t_unc", "gap", "monotonicity", "auc_uncertainty"]
    lines = [",".join(header)]
    for e in events:
        lines.append(
            ",".join(
                [
                    str(e["transform"]),
                    f"{float(e['sample_id']):.0f}",
                    str(e["model"]),
                    f"{float(e['t_flip']):.8f}" if not np.isnan(float(e["t_flip"])) else "nan",
                    f"{float(e['t_unc']):.8f}" if not np.isnan(float(e["t_unc"])) else "nan",
                    f"{float(e['gap']):.8f}" if not np.isnan(float(e["gap"])) else "nan",
                    f"{float(e['monotonicity']):.8f}",
                    f"{float(e['auc_uncertainty']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_csv(path: Path, events: list[dict[str, float | str]]) -> None:
    header = [
        "transform",
        "model",
        "mean_t_flip",
        "mean_t_unc",
        "mean_gap",
        "frac_early_warning",
        "mean_monotonicity",
        "mean_auc_uncertainty",
        "n_valid_gap",
    ]
    lines = [",".join(header)]
    transforms = sorted({str(e["transform"]) for e in events})
    models = ["dnn", "gp"]
    for tr in transforms:
        for m in models:
            rows = [e for e in events if e["transform"] == tr and e["model"] == m]
            t_flip = np.array([float(e["t_flip"]) for e in rows], dtype=float)
            t_unc = np.array([float(e["t_unc"]) for e in rows], dtype=float)
            gap = np.array([float(e["gap"]) for e in rows], dtype=float)
            mono = np.array([float(e["monotonicity"]) for e in rows], dtype=float)
            auc = np.array([float(e["auc_uncertainty"]) for e in rows], dtype=float)
            valid_gap = ~np.isnan(gap)
            frac_early = float(np.mean(gap[valid_gap] > 0.0)) if np.any(valid_gap) else np.nan
            line = [
                tr,
                m,
                f"{np.nanmean(t_flip):.8f}",
                f"{np.nanmean(t_unc):.8f}",
                f"{np.nanmean(gap):.8f}",
                f"{frac_early:.8f}" if not np.isnan(frac_early) else "nan",
                f"{np.mean(mono):.8f}",
                f"{np.mean(auc):.8f}",
                str(int(np.sum(valid_gap))),
            ]
            lines.append(",".join(line))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trajectories(path: Path, packed_by_transform: dict[str, dict[str, np.ndarray]]) -> None:
    transforms = list(packed_by_transform.keys())
    fig, axes = plt.subplots(1, len(transforms), figsize=(5.0 * len(transforms), 3.8))
    if len(transforms) == 1:
        axes = [axes]
    for ax, tr in zip(axes, transforms):
        d = packed_by_transform[tr]
        t = d["t_grid"]
        ax.plot(t, d["dnn_entropy_mean"], "-o", color="#1F4E79", label="DNN entropy", linewidth=2, markersize=4)
        ax.plot(t, d["gp_entropy_mean"], "-o", color="#A61C3C", label="GP entropy", linewidth=2, markersize=4)
        ax.plot(t, d["dnn_one_minus_max_mean"], "--", color="#4C78A8", label="DNN 1-max", linewidth=1.8)
        ax.plot(t, d["gp_one_minus_max_mean"], "--", color="#C4455A", label="GP 1-max", linewidth=1.8)
        ax.set_xlabel("Normalized transform strength t")
        ax.set_ylabel("Uncertainty")
        ax.set_title(tr.capitalize())
        ax.grid(True)
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap_distribution(path: Path, events: list[dict[str, float | str]]) -> None:
    transforms = sorted({str(e["transform"]) for e in events})
    fig, axes = plt.subplots(1, len(transforms), figsize=(5.0 * len(transforms), 3.8))
    if len(transforms) == 1:
        axes = [axes]
    bins = np.linspace(-1.0, 1.0, 17)
    for ax, tr in zip(axes, transforms):
        dnn_gap = np.array([float(e["gap"]) for e in events if e["transform"] == tr and e["model"] == "dnn"], dtype=float)
        gp_gap = np.array([float(e["gap"]) for e in events if e["transform"] == tr and e["model"] == "gp"], dtype=float)
        dnn_gap = dnn_gap[~np.isnan(dnn_gap)]
        gp_gap = gp_gap[~np.isnan(gp_gap)]
        ax.hist(dnn_gap, bins=bins, alpha=0.55, color="#1F4E79", label="DNN", density=True)
        ax.hist(gp_gap, bins=bins, alpha=0.55, color="#A61C3C", label="GP", density=True)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Early-warning gap (t_flip - t_unc)")
        ax.set_ylabel("Density")
        ax.set_title(tr.capitalize())
        ax.grid(True)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_fig = out_dir / args.traj_figure
    gap_fig = out_dir / args.gap_figure
    summary_csv = out_dir / args.summary_csv
    events_csv = out_dir / args.events_csv

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
    base_images, _ = sample_balanced_test_subset(test_set, n_per_class=args.n_per_class, seed=args.seed + 2)
    n_samples = base_images.shape[0]

    t_grid = np.linspace(0.0, 1.0, args.steps)
    trajs = {
        "rotation": build_rotation_traj(base_images, t_grid, args.rotation_max_deg),
        "translation": build_translation_traj(base_images, t_grid, args.translation_max_px),
        "scale": build_scale_traj(base_images, t_grid, args.scale_min),
    }

    packed_by_transform: dict[str, dict[str, np.ndarray]] = {}
    all_events: list[dict[str, float | str]] = []
    for i, (name, traj_images) in enumerate(trajs.items()):
        print(f"Evaluating transform trajectory: {name}")
        packed, events = evaluate_transform(
            name=name,
            traj_images=traj_images,
            n_steps=args.steps,
            n_samples=n_samples,
            t_grid=t_grid,
            model=model,
            post_prec=post_prec,
            device=device,
            gp_mc_samples=args.gp_mc_samples,
            seed=args.seed + 20 + i,
            rise_frac=args.unc_rise_fraction,
        )
        packed_by_transform[name] = packed
        all_events.extend(events)

    save_events_csv(events_csv, all_events)
    save_summary_csv(summary_csv, all_events)
    plot_trajectories(traj_fig, packed_by_transform)
    plot_gap_distribution(gap_fig, all_events)

    print(f"Saved figure: {traj_fig}")
    print(f"Saved figure: {gap_fig}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved events: {events_csv}")


if __name__ == "__main__":
    main()

