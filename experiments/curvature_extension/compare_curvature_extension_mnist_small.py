#!/usr/bin/env python3
"""Small MNIST experiment: original DNN2GP vs curvature-weighted extension.

This script is intentionally lightweight:
- uses an existing MNIST LeNet checkpoint
- computes Laplace diagonal precision on a small train subset
- evaluates on a small test subset
- compares kernels and accuracies for:
  1) MAP classifier
  2) DNN2GP (original Jacobian features)
  3) curvature-weighted extension (phi = Lambda(x)^{1/2} J(x))
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import Dataset
from dnn2gp import compute_laplace
from neural_networks import LeNet5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST small comparison: DNN2GP vs curvature extension.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--output-dir", type=str, default="results/curvature_extension")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--laplace-train-size", type=int, default=800)
    parser.add_argument("--eval-size", type=int, default=220)
    parser.add_argument("--kernel-size", type=int, default=60)
    parser.add_argument("--mc-samples", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-prec", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    req = device_arg.lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = LeNet5(input_channels=1, dims=28, num_classes=10).to(device=device, dtype=torch.double)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_subsets(data_dir: str, laplace_train_size: int, eval_size: int, seed: int) -> tuple[Subset, Subset]:
    ds = Dataset("mnist", data_folder=data_dir)
    train_set = ds.train_set
    test_set = ds.test_set
    gen = torch.Generator().manual_seed(seed)

    tr_idx = torch.randperm(len(train_set), generator=gen)[: min(laplace_train_size, len(train_set))].tolist()
    te_idx = torch.randperm(len(test_set), generator=gen)[: min(eval_size, len(test_set))].tolist()
    return Subset(train_set, tr_idx), Subset(test_set, te_idx)


def gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads, dim=0)


def jacobian_and_logits(model: torch.nn.Module, x_single: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(x_single).squeeze(0)
    jac_rows = []
    for c in range(logits.numel()):
        retain = c < logits.numel() - 1
        logits[c].backward(retain_graph=retain)
        jac_rows.append(gradient_vector(model))
        model.zero_grad(set_to_none=True)
    return logits.detach(), torch.stack(jac_rows, dim=0).detach()


def matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mat = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def mc_probs_from_diag_logit_var(
    mean_logits: torch.Tensor,
    var_logits: torch.Tensor,
    mc_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    eps = torch.randn(
        (mc_samples, mean_logits.numel()),
        dtype=mean_logits.dtype,
        device=mean_logits.device,
        generator=generator,
    )
    samples = mean_logits.unsqueeze(0) + eps * torch.sqrt(var_logits.clamp_min(1e-10)).unsqueeze(0)
    probs = torch.softmax(samples, dim=1)
    probs_mean = probs.mean(dim=0)
    entropy = float((-probs_mean * torch.log(probs_mean.clamp_min(1e-12))).sum().item())
    return probs_mean, entropy


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(kernel), a_min=1e-12, a_max=None))
    return kernel / (d[:, None] * d[None, :])


def prior_precision_tag(value: float) -> str:
    return format(value, ".3g").replace(".", "p").replace("-", "m")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    exp_dir = output_dir / "mnist_curvature_extension"
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    model = load_model(checkpoint_path=checkpoint_path, device=device)
    train_subset, test_subset = load_subsets(
        data_dir=args.data_dir,
        laplace_train_size=args.laplace_train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    pp_tag = prior_precision_tag(args.prior_prec)
    prec_cache = exp_dir / f"laplace_post_prec_train{len(train_subset)}_pp{pp_tag}.pt"
    if prec_cache.exists():
        post_prec = torch.load(prec_cache, map_location=device).to(device=device, dtype=torch.double)
    else:
        train_loader_lap = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        post_prec = compute_laplace(
            model=model,
            train_loader=train_loader_lap,
            prior_prec=args.prior_prec,
            device=device,
        ).detach()
        torch.save(post_prec.detach().cpu(), prec_cache)

    inv_post_prec = torch.reciprocal(post_prec.clamp_min(1e-12))
    eval_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 999)

    y_true = []
    pred_map, pred_dnn2gp, pred_curv = [], [], []
    entropy_dnn2gp, entropy_curv = [], []
    lambda_trace = []
    kernel_j, kernel_j_curv = [], []
    kernel_labels = []

    for x, y in tqdm(eval_loader, desc="MNIST DNN2GP/curvature eval"):
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.long)
        logits, jac = jacobian_and_logits(model, x)
        p_map = torch.softmax(logits, dim=0)
        lam = torch.diag(p_map) - torch.outer(p_map, p_map)
        sqrt_lam = matrix_sqrt_psd(lam)

        jac_curv = sqrt_lam @ jac
        cov_logits = torch.einsum("kp,p,mp->km", jac, inv_post_prec, jac)
        # Keep "original" aligned with src/dnn2gp.py, which applies Lambda on both sides.
        var_orig = torch.diag(lam @ cov_logits @ lam).clamp_min(1e-10)
        var_curv = torch.diag(sqrt_lam @ cov_logits @ sqrt_lam).clamp_min(1e-10)

        probs_orig, ent_orig = mc_probs_from_diag_logit_var(
            mean_logits=logits,
            var_logits=var_orig,
            mc_samples=args.mc_samples,
            generator=generator,
        )
        probs_curv, ent_curv = mc_probs_from_diag_logit_var(
            mean_logits=logits,
            var_logits=var_curv,
            mc_samples=args.mc_samples,
            generator=generator,
        )

        y_true.append(int(y.item()))
        pred_map.append(int(torch.argmax(p_map).item()))
        pred_dnn2gp.append(int(torch.argmax(probs_orig).item()))
        pred_curv.append(int(torch.argmax(probs_curv).item()))
        entropy_dnn2gp.append(ent_orig)
        entropy_curv.append(ent_curv)
        lambda_trace.append(float(torch.trace(lam).item()))

        if len(kernel_j) < args.kernel_size:
            kernel_j.append(jac.detach().cpu().float().numpy())
            kernel_j_curv.append(jac_curv.detach().cpu().float().numpy())
            kernel_labels.append(int(y.item()))

    y_true_np = np.asarray(y_true)
    pred_map_np = np.asarray(pred_map)
    pred_dnn2gp_np = np.asarray(pred_dnn2gp)
    pred_curv_np = np.asarray(pred_curv)
    entropy_dnn2gp_np = np.asarray(entropy_dnn2gp)
    entropy_curv_np = np.asarray(entropy_curv)
    lambda_trace_np = np.asarray(lambda_trace)

    acc_map = float(np.mean(pred_map_np == y_true_np))
    acc_dnn2gp = float(np.mean(pred_dnn2gp_np == y_true_np))
    acc_curv = float(np.mean(pred_curv_np == y_true_np))

    J = np.stack(kernel_j, axis=0)
    Jc = np.stack(kernel_j_curv, axis=0)
    kernel_labels_np = np.asarray(kernel_labels, dtype=np.int64)
    sort_idx = np.argsort(kernel_labels_np, kind="stable")
    J = J[sort_idx]
    Jc = Jc[sort_idx]
    kernel_labels_np = kernel_labels_np[sort_idx]
    K_orig = np.einsum("ikp,jkp->ij", J, J, optimize=True)
    K_curv = np.einsum("ikp,jkp->ij", Jc, Jc, optimize=True)
    K_orig_n = normalize_kernel(K_orig)
    K_curv_n = normalize_kernel(K_curv)
    K_diff_n = K_curv_n - K_orig_n

    tri = np.triu_indices_from(K_orig_n, k=1)
    kernel_corr = float(np.corrcoef(K_orig_n[tri], K_curv_n[tri])[0, 1]) if tri[0].size > 1 else float("nan")

    np.save(exp_dir / "kernel_original.npy", K_orig)
    np.save(exp_dir / "kernel_curvature.npy", K_curv)
    np.save(exp_dir / "kernel_original_normalized.npy", K_orig_n)
    np.save(exp_dir / "kernel_curvature_normalized.npy", K_curv_n)
    np.savez(
        exp_dir / "predictions_and_metrics_arrays.npz",
        y_true=y_true_np,
        pred_map=pred_map_np,
        pred_dnn2gp=pred_dnn2gp_np,
        pred_curvature=pred_curv_np,
        entropy_dnn2gp=entropy_dnn2gp_np,
        entropy_curvature=entropy_curv_np,
        lambda_trace=lambda_trace_np,
        kernel_labels=kernel_labels_np,
    )

    fig, axs = plt.subplots(2, 3, figsize=(15.8, 9.0))
    im0 = axs[0, 0].imshow(K_orig_n, cmap="magma", aspect="auto")
    axs[0, 0].set_title("Original DNN2GP kernel (normalized, sorted by class)")
    axs[0, 0].set_xlabel("sample index")
    axs[0, 0].set_ylabel("sample index")
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(K_curv_n, cmap="magma", aspect="auto")
    axs[0, 1].set_title("Curvature-weighted kernel (normalized, sorted by class)")
    axs[0, 1].set_xlabel("sample index")
    axs[0, 1].set_ylabel("sample index")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    vmax = float(np.max(np.abs(K_diff_n)))
    im2 = axs[0, 2].imshow(K_diff_n, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axs[0, 2].set_title("Kernel difference (curv - orig)")
    axs[0, 2].set_xlabel("sample index")
    axs[0, 2].set_ylabel("sample index")
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)
    boundaries = np.where(np.diff(kernel_labels_np) != 0)[0]
    for b in boundaries:
        for ax in [axs[0, 0], axs[0, 1], axs[0, 2]]:
            ax.axhline(b + 0.5, color="white", lw=0.4, alpha=0.7)
            ax.axvline(b + 0.5, color="white", lw=0.4, alpha=0.7)

    axs[1, 0].bar(
        ["MAP", "DNN2GP", "Curvature"],
        [acc_map, acc_dnn2gp, acc_curv],
        color=["#1f77b4", "#2ca02c", "#d62728"],
    )
    axs[1, 0].set_ylim(0.0, 1.0)
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].set_title("Test subset accuracy")
    axs[1, 0].grid(alpha=0.2, axis="y")

    log_ent_dnn2gp = np.log10(np.clip(entropy_dnn2gp_np, a_min=1e-12, a_max=None))
    log_ent_curv = np.log10(np.clip(entropy_curv_np, a_min=1e-12, a_max=None))
    axs[1, 1].hist(log_ent_dnn2gp, bins=26, alpha=0.6, label="DNN2GP", color="#2ca02c", density=True)
    axs[1, 1].hist(log_ent_curv, bins=26, alpha=0.6, label="Curvature", color="#d62728", density=True)
    axs[1, 1].set_title("Predictive entropy distribution (log10)")
    axs[1, 1].set_xlabel("log10(entropy)")
    axs[1, 1].set_ylabel("Density")
    axs[1, 1].legend(loc="best")
    axs[1, 1].grid(alpha=0.2)

    text = "\n".join(
        [
            f"eval size: {len(y_true_np)}",
            f"kernel size: {len(kernel_j)}",
            "dataset: MNIST 10 classes",
            f"MC samples: {args.mc_samples}",
            f"acc MAP: {acc_map:.4f}",
            f"acc DNN2GP: {acc_dnn2gp:.4f}",
            f"acc Curvature: {acc_curv:.4f}",
            f"kernel corr (upper tri): {kernel_corr:.4f}",
            f"entropy mean DNN2GP/Curv: {entropy_dnn2gp_np.mean():.4f} / {entropy_curv_np.mean():.4f}",
            f"entropy median DNN2GP/Curv: {np.median(entropy_dnn2gp_np):.4f} / {np.median(entropy_curv_np):.4e}",
            f"Lambda trace mean/std: {lambda_trace_np.mean():.4f} / {lambda_trace_np.std():.4f}",
        ]
    )
    axs[1, 2].axis("off")
    axs[1, 2].text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=10.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.5"),
    )
    axs[1, 2].set_title("Summary metrics")

    fig.suptitle("MNIST small: DNN2GP vs curvature-weighted extension", y=1.01, fontsize=13)
    fig.tight_layout()
    fig_path = exp_dir / "mnist_small_dnn2gp_vs_curvature_extension.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "dataset": "MNIST (small subset)",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "laplace_train_size": len(train_subset),
        "eval_size": int(len(y_true_np)),
        "kernel_size": int(len(kernel_j)),
        "mc_samples": int(args.mc_samples),
        "accuracy_map": acc_map,
        "accuracy_dnn2gp": acc_dnn2gp,
        "accuracy_curvature_extension": acc_curv,
        "kernel_corr_upper_tri_normalized": kernel_corr,
        "kernel_trace_original": float(np.trace(K_orig)),
        "kernel_trace_curvature": float(np.trace(K_curv)),
        "kernel_fro_original": float(np.linalg.norm(K_orig, ord="fro")),
        "kernel_fro_curvature": float(np.linalg.norm(K_curv, ord="fro")),
        "kernel_fro_diff_normalized": float(np.linalg.norm(K_diff_n, ord="fro")),
        "entropy_mean_dnn2gp": float(np.mean(entropy_dnn2gp_np)),
        "entropy_mean_curvature": float(np.mean(entropy_curv_np)),
        "lambda_trace_mean": float(np.mean(lambda_trace_np)),
        "lambda_trace_std": float(np.std(lambda_trace_np)),
        "note": (
            "Unlike scalar Gaussian regression, Lambda(x)=diag(p)-pp^T in multiclass classification "
            "depends on x through p(x), so curvature weighting is nontrivial here."
        ),
        "figure_path": str(fig_path),
    }
    metrics_path = exp_dir / "mnist_small_curvature_extension_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Accuracies | MAP: "
        f"{acc_map:.4f}, DNN2GP: {acc_dnn2gp:.4f}, Curvature: {acc_curv:.4f}"
    )


if __name__ == "__main__":
    main()
