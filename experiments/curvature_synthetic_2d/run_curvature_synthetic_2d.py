#!/usr/bin/env python3
"""Synthetic 2D binary classification to probe curvature-weighted DNN2GP behavior.

This experiment is intentionally small and reuses existing repository pieces:
- MAP training with a compact MLP
- Laplace diagonal posterior precision from src/dnn2gp.py
- DNN2GP-style uncertainty proxy using Lambda around MAP logits
- Curvature-weighted variant with sqrt(Lambda)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dnn2gp import compute_laplace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature synthetic 2D binary experiment.")
    parser.add_argument("--output-dir", type=str, default="results/curvature_synthetic_2d")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=2500)
    parser.add_argument("--laplace-train-size", type=int, default=1200)

    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--grid-min", type=float, default=-3.0)
    parser.add_argument("--grid-max", type=float, default=3.0)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--prior-prec", type=float, default=1.0)

    parser.add_argument("--mc-samples", type=int, default=80)
    parser.add_argument("--kernel-size", type=int, default=220)
    parser.add_argument("--center-radius", type=float, default=1.5)
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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def g_fn(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 0.8 * np.sin(2.0 * x[:, 1])


def tau_fn(x: np.ndarray) -> np.ndarray:
    return 0.35 + 1.8 * np.exp(-((x[:, 0] ** 2 + x[:, 1] ** 2) / (1.2**2)))


def true_prob_fn(x: np.ndarray) -> np.ndarray:
    logits = g_fn(x) / tau_fn(x)
    return sigmoid_np(logits)


def sample_synthetic_binary_2d(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(low=-3.0, high=3.0, size=(n, 2)).astype(np.float64)
    p = true_prob_fn(x)
    y = rng.binomial(1, p, size=(n,)).astype(np.int64)
    return x, y, p


class BinaryMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, n_hidden: int = 2, activation: str = "tanh"):
        super().__init__()
        self.fc_in = nn.Linear(2, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(max(0, n_hidden - 1))])
        self.fc_out = nn.Linear(hidden_size, 2)
        self.activation_name = activation

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(x)
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc_in(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.fc_out(x)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_map_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = int(xb.shape[0])
            loss_sum += float(loss.item()) * bs
            n_items += bs

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            train_acc = evaluate_map_accuracy(model, train_loader, device)
            test_acc = evaluate_map_accuracy(model, test_loader, device)
            print(
                f"epoch {epoch:03d} | loss {loss_sum / max(1, n_items):.4f} | "
                f"train acc {train_acc:.4f} | test acc {test_acc:.4f}"
            )


@torch.no_grad()
def evaluate_map_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_total = 0
    n_correct = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)
        pred = model(xb).argmax(dim=1)
        n_correct += int((pred == yb).sum().item())
        n_total += int(yb.numel())
    return n_correct / max(1, n_total)


def gradient_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads, dim=0)


def jacobian_and_logits(model: nn.Module, x_single: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    samples = mean_logits.unsqueeze(0) + eps * torch.sqrt(var_logits.clamp_min(1e-12)).unsqueeze(0)
    probs = torch.softmax(samples, dim=1)
    probs_mean = probs.mean(dim=0)
    entropy = float((-probs_mean * torch.log(probs_mean.clamp_min(1e-12))).sum().item())
    return probs_mean, entropy


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(kernel), a_min=1e-12, a_max=None))
    return kernel / (d[:, None] * d[None, :])


def evaluate_methods_on_points(
    model: nn.Module,
    inv_post_prec: torch.Tensor,
    x_np: np.ndarray,
    y_np: np.ndarray | None,
    mc_samples: int,
    seed: int,
    kernel_size: int,
    collect_kernel: bool,
) -> dict[str, np.ndarray | float]:
    device = inv_post_prec.device
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)

    prob_map_list, pred_map_list = [], []
    prob_orig_list, pred_orig_list, ent_orig_list = [], [], []
    prob_curv_list, pred_curv_list, ent_curv_list = [], [], []
    lambda_trace_list = []

    kernel_j, kernel_j_curv, kernel_labels = [], [], []

    iterator = tqdm(range(x_np.shape[0]), desc="Evaluating MAP/DNN2GP/curvature", leave=False)
    for i in iterator:
        x_single = torch.from_numpy(x_np[i : i + 1]).to(device=device, dtype=torch.double)

        logits, jac = jacobian_and_logits(model, x_single)
        p_map = torch.softmax(logits, dim=0)
        lam = torch.diag(p_map) - torch.outer(p_map, p_map)
        sqrt_lam = matrix_sqrt_psd(lam)

        cov_logits = torch.einsum("kp,p,mp->km", jac, inv_post_prec, jac)
        var_orig = torch.diag(lam @ cov_logits @ lam).clamp_min(1e-12)
        var_curv = torch.diag(sqrt_lam @ cov_logits @ sqrt_lam).clamp_min(1e-12)

        probs_orig, ent_orig = mc_probs_from_diag_logit_var(logits, var_orig, mc_samples, generator)
        probs_curv, ent_curv = mc_probs_from_diag_logit_var(logits, var_curv, mc_samples, generator)

        prob_map_list.append(float(p_map[1].item()))
        pred_map_list.append(int(torch.argmax(p_map).item()))

        prob_orig_list.append(float(probs_orig[1].item()))
        pred_orig_list.append(int(torch.argmax(probs_orig).item()))
        ent_orig_list.append(ent_orig)

        prob_curv_list.append(float(probs_curv[1].item()))
        pred_curv_list.append(int(torch.argmax(probs_curv).item()))
        ent_curv_list.append(ent_curv)

        lambda_trace_list.append(float(torch.trace(lam).item()))

        if collect_kernel and len(kernel_j) < kernel_size:
            kernel_j.append(jac.detach().cpu().float().numpy())
            kernel_j_curv.append((sqrt_lam @ jac).detach().cpu().float().numpy())
            if y_np is None:
                kernel_labels.append(-1)
            else:
                kernel_labels.append(int(y_np[i]))

    out: dict[str, np.ndarray | float] = {
        "prob_map": np.asarray(prob_map_list, dtype=np.float64),
        "pred_map": np.asarray(pred_map_list, dtype=np.int64),
        "prob_dnn2gp": np.asarray(prob_orig_list, dtype=np.float64),
        "pred_dnn2gp": np.asarray(pred_orig_list, dtype=np.int64),
        "entropy_dnn2gp": np.asarray(ent_orig_list, dtype=np.float64),
        "prob_curvature": np.asarray(prob_curv_list, dtype=np.float64),
        "pred_curvature": np.asarray(pred_curv_list, dtype=np.int64),
        "entropy_curvature": np.asarray(ent_curv_list, dtype=np.float64),
        "lambda_trace": np.asarray(lambda_trace_list, dtype=np.float64),
    }

    if y_np is not None:
        out["accuracy_map"] = float(np.mean(out["pred_map"] == y_np))
        out["accuracy_dnn2gp"] = float(np.mean(out["pred_dnn2gp"] == y_np))
        out["accuracy_curvature"] = float(np.mean(out["pred_curvature"] == y_np))

    if collect_kernel and len(kernel_j) > 1:
        J = np.stack(kernel_j, axis=0)
        Jc = np.stack(kernel_j_curv, axis=0)
        labels = np.asarray(kernel_labels, dtype=np.int64)
        if y_np is not None:
            sort_idx = np.argsort(labels, kind="stable")
            J = J[sort_idx]
            Jc = Jc[sort_idx]
            labels = labels[sort_idx]

        K_orig = np.einsum("ikp,jkp->ij", J, J, optimize=True)
        K_curv = np.einsum("ikp,jkp->ij", Jc, Jc, optimize=True)
        K_orig_n = normalize_kernel(K_orig)
        K_curv_n = normalize_kernel(K_curv)

        tri = np.triu_indices_from(K_orig_n, k=1)
        kernel_corr = float(np.corrcoef(K_orig_n[tri], K_curv_n[tri])[0, 1]) if tri[0].size > 1 else float("nan")

        out["kernel_original"] = K_orig
        out["kernel_curvature"] = K_curv
        out["kernel_original_normalized"] = K_orig_n
        out["kernel_curvature_normalized"] = K_curv_n
        out["kernel_labels"] = labels
        out["kernel_corr"] = kernel_corr

    return out


def save_dataset_figure(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    true_prob: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    center_radius: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(
        true_prob,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    sc = ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=10, alpha=0.75, edgecolors="none")
    _ = sc
    ax.add_patch(plt.Circle((0.0, 0.0), center_radius, color="white", fill=False, lw=1.1, ls="--", alpha=0.8))
    ax.set_title("Synthetic 2D data: train samples over true probability p(y=1|x)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("true p(y=1|x)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_three_panel_map(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    title1: str,
    title2: str,
    diff_title: str,
    cmap_main: str,
    center_radius: float,
    dpi: int,
) -> None:
    diff = z2 - z1
    vdiff = float(np.max(np.abs(diff)))

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9))
    im0 = axs[0].imshow(
        z1,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap=cmap_main,
        aspect="equal",
    )
    axs[0].set_title(title1)
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        z2,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap=cmap_main,
        aspect="equal",
    )
    axs[1].set_title(title2)
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(
        diff,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="coolwarm",
        vmin=-vdiff,
        vmax=vdiff,
        aspect="equal",
    )
    axs[2].set_title(diff_title)
    axs[2].set_xlabel("x1")
    axs[2].set_ylabel("x2")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.add_patch(plt.Circle((0.0, 0.0), center_radius, color="white", fill=False, lw=1.0, ls="--", alpha=0.8))
        ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_lambda_map(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    true_lambda_scalar: np.ndarray,
    model_lambda_trace: np.ndarray,
    center_radius: float,
    dpi: int,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.8))

    im0 = axs[0].imshow(
        true_lambda_scalar,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="magma",
        aspect="equal",
    )
    axs[0].set_title("True curvature proxy p(1-p)")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        model_lambda_trace,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="magma",
        aspect="equal",
    )
    axs[1].set_title("MAP Lambda trace tr(diag(p)-pp^T)")
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.add_patch(plt.Circle((0.0, 0.0), center_radius, color="white", fill=False, lw=1.0, ls="--", alpha=0.8))
        ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    rng_train = np.random.default_rng(args.seed)
    rng_test = np.random.default_rng(args.seed + 1)

    x_train, y_train, p_train_true = sample_synthetic_binary_2d(args.n_train, rng_train)
    x_test, y_test, p_test_true = sample_synthetic_binary_2d(args.n_test, rng_test)

    train_loader = make_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size=512, shuffle=False)

    model = BinaryMLP(hidden_size=args.hidden_size, n_hidden=args.n_hidden, activation=args.activation).to(
        device=device, dtype=torch.float32
    )
    train_map_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_acc_map = evaluate_map_accuracy(model, make_loader(x_train, y_train, batch_size=512, shuffle=False), device)
    test_acc_map = evaluate_map_accuracy(model, test_loader, device)

    # Laplace precision for DNN2GP-style posterior around MAP.
    laplace_size = min(args.laplace_train_size, x_train.shape[0])
    lap_idx = np.random.default_rng(args.seed + 2).choice(x_train.shape[0], size=laplace_size, replace=False)
    lap_loader = make_loader(x_train[lap_idx], y_train[lap_idx], batch_size=args.laplace_batch_size, shuffle=False)

    model_double = copy.deepcopy(model).to(device=device, dtype=torch.double).eval()
    post_prec = compute_laplace(model=model_double, train_loader=lap_loader, prior_prec=args.prior_prec, device=device)
    inv_post_prec = torch.reciprocal(post_prec.clamp_min(1e-12))

    # Test-set comparison metrics.
    test_eval = evaluate_methods_on_points(
        model=model_double,
        inv_post_prec=inv_post_prec,
        x_np=x_test,
        y_np=y_test,
        mc_samples=args.mc_samples,
        seed=args.seed + 100,
        kernel_size=args.kernel_size,
        collect_kernel=True,
    )

    grid_lin = np.linspace(args.grid_min, args.grid_max, args.grid_size)
    xx, yy = np.meshgrid(grid_lin, grid_lin)
    x_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    p_true_grid = true_prob_fn(x_grid)

    grid_eval = evaluate_methods_on_points(
        model=model_double,
        inv_post_prec=inv_post_prec,
        x_np=x_grid,
        y_np=None,
        mc_samples=args.mc_samples,
        seed=args.seed + 200,
        kernel_size=0,
        collect_kernel=False,
    )

    prob_dnn2gp_grid = np.asarray(grid_eval["prob_dnn2gp"]).reshape(args.grid_size, args.grid_size)
    prob_curv_grid = np.asarray(grid_eval["prob_curvature"]).reshape(args.grid_size, args.grid_size)
    ent_dnn2gp_grid = np.asarray(grid_eval["entropy_dnn2gp"]).reshape(args.grid_size, args.grid_size)
    ent_curv_grid = np.asarray(grid_eval["entropy_curvature"]).reshape(args.grid_size, args.grid_size)
    lambda_trace_grid = np.asarray(grid_eval["lambda_trace"]).reshape(args.grid_size, args.grid_size)
    p_true_grid_img = p_true_grid.reshape(args.grid_size, args.grid_size)

    center_mask = (xx**2 + yy**2) <= (args.center_radius**2)
    outer_mask = ~center_mask

    ent_center_dnn2gp = float(np.mean(ent_dnn2gp_grid[center_mask]))
    ent_outer_dnn2gp = float(np.mean(ent_dnn2gp_grid[outer_mask]))
    ent_center_curv = float(np.mean(ent_curv_grid[center_mask]))
    ent_outer_curv = float(np.mean(ent_curv_grid[outer_mask]))

    # Save figures.
    fig1 = out_dir / "dataset_true_probability_and_train_points.png"
    save_dataset_figure(
        out_path=fig1,
        xx=xx,
        yy=yy,
        true_prob=p_true_grid_img,
        x_train=x_train,
        y_train=y_train,
        center_radius=args.center_radius,
        dpi=args.dpi,
    )

    fig2 = out_dir / "predictive_probability_dnn2gp_vs_curvature.png"
    save_three_panel_map(
        out_path=fig2,
        xx=xx,
        yy=yy,
        z1=prob_dnn2gp_grid,
        z2=prob_curv_grid,
        title1="DNN2GP predictive p(y=1|x)",
        title2="Curvature-weighted predictive p(y=1|x)",
        diff_title="Difference (curvature - DNN2GP)",
        cmap_main="viridis",
        center_radius=args.center_radius,
        dpi=args.dpi,
    )

    fig3 = out_dir / "predictive_entropy_dnn2gp_vs_curvature.png"
    save_three_panel_map(
        out_path=fig3,
        xx=xx,
        yy=yy,
        z1=ent_dnn2gp_grid,
        z2=ent_curv_grid,
        title1="DNN2GP predictive entropy",
        title2="Curvature-weighted predictive entropy",
        diff_title="Entropy difference (curvature - DNN2GP)",
        cmap_main="magma",
        center_radius=args.center_radius,
        dpi=args.dpi,
    )

    fig4 = out_dir / "lambda_variation_map.png"
    save_lambda_map(
        out_path=fig4,
        xx=xx,
        yy=yy,
        true_lambda_scalar=(p_true_grid_img * (1.0 - p_true_grid_img)),
        model_lambda_trace=lambda_trace_grid,
        center_radius=args.center_radius,
        dpi=args.dpi,
    )

    # Save kernel arrays and evaluation arrays.
    np.save(out_dir / "kernel_original.npy", np.asarray(test_eval["kernel_original"]))
    np.save(out_dir / "kernel_curvature.npy", np.asarray(test_eval["kernel_curvature"]))
    np.save(out_dir / "kernel_original_normalized.npy", np.asarray(test_eval["kernel_original_normalized"]))
    np.save(out_dir / "kernel_curvature_normalized.npy", np.asarray(test_eval["kernel_curvature_normalized"]))

    np.savez(
        out_dir / "synthetic_2d_eval_arrays.npz",
        x_train=x_train,
        y_train=y_train,
        p_train_true=p_train_true,
        x_test=x_test,
        y_test=y_test,
        p_test_true=p_test_true,
        test_prob_dnn2gp=np.asarray(test_eval["prob_dnn2gp"]),
        test_prob_curvature=np.asarray(test_eval["prob_curvature"]),
        test_entropy_dnn2gp=np.asarray(test_eval["entropy_dnn2gp"]),
        test_entropy_curvature=np.asarray(test_eval["entropy_curvature"]),
        test_pred_map=np.asarray(test_eval["pred_map"]),
        test_pred_dnn2gp=np.asarray(test_eval["pred_dnn2gp"]),
        test_pred_curvature=np.asarray(test_eval["pred_curvature"]),
        xx=xx,
        yy=yy,
        center_mask=center_mask,
        prob_dnn2gp_grid=prob_dnn2gp_grid,
        prob_curvature_grid=prob_curv_grid,
        entropy_dnn2gp_grid=ent_dnn2gp_grid,
        entropy_curvature_grid=ent_curv_grid,
        lambda_trace_grid=lambda_trace_grid,
        true_prob_grid=p_true_grid_img,
    )

    acc_dnn2gp = float(test_eval["accuracy_dnn2gp"])
    acc_curv = float(test_eval["accuracy_curvature"])

    if acc_curv > acc_dnn2gp + 0.005:
        acc_note = "curvature helps accuracy"
    elif acc_dnn2gp > acc_curv + 0.005:
        acc_note = "curvature hurts accuracy"
    else:
        acc_note = "accuracy is similar"

    if ent_center_curv < ent_center_dnn2gp and ent_outer_curv <= ent_outer_dnn2gp:
        uncertainty_note = "curvature is broadly more confident"
    elif ent_center_curv > ent_center_dnn2gp and ent_outer_curv <= ent_outer_dnn2gp:
        uncertainty_note = "curvature raises uncertainty mainly in central ambiguous region"
    elif ent_center_curv > ent_center_dnn2gp and ent_outer_curv > ent_outer_dnn2gp:
        uncertainty_note = "curvature is broadly more uncertain"
    else:
        uncertainty_note = "curvature mostly reshapes uncertainty without simple global trend"

    summary = {
        "experiment": "curvature_synthetic_2d_binary",
        "device": str(device),
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "laplace_train_size": int(laplace_size),
        "grid_size": int(args.grid_size),
        "center_radius": float(args.center_radius),
        "map_train_accuracy": float(train_acc_map),
        "map_test_accuracy": float(test_acc_map),
        "dnn2gp_test_accuracy": acc_dnn2gp,
        "curvature_test_accuracy": acc_curv,
        "entropy_center_dnn2gp": ent_center_dnn2gp,
        "entropy_outer_dnn2gp": ent_outer_dnn2gp,
        "entropy_center_curvature": ent_center_curv,
        "entropy_outer_curvature": ent_outer_curv,
        "entropy_center_diff_curvature_minus_dnn2gp": float(ent_center_curv - ent_center_dnn2gp),
        "entropy_outer_diff_curvature_minus_dnn2gp": float(ent_outer_curv - ent_outer_dnn2gp),
        "kernel_corr_upper_tri_normalized": float(test_eval["kernel_corr"]),
        "lambda_trace_grid_mean": float(np.mean(lambda_trace_grid)),
        "lambda_trace_grid_std": float(np.std(lambda_trace_grid)),
        "true_lambda_scalar_grid_mean": float(np.mean(p_true_grid_img * (1.0 - p_true_grid_img))),
        "true_lambda_scalar_grid_std": float(np.std(p_true_grid_img * (1.0 - p_true_grid_img))),
        "quick_take_accuracy": acc_note,
        "quick_take_uncertainty": uncertainty_note,
        "figures": {
            "dataset": str(fig1),
            "probability_comparison": str(fig2),
            "entropy_comparison": str(fig3),
            "lambda_map": str(fig4),
        },
    }

    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figures and arrays in: {out_dir}")
    print(f"Saved summary: {summary_path}")
    print(
        "Accuracies | MAP(test): "
        f"{test_acc_map:.4f}, DNN2GP: {acc_dnn2gp:.4f}, Curvature: {acc_curv:.4f}"
    )
    print(
        "Entropy center/outer (DNN2GP vs Curvature) | "
        f"center: {ent_center_dnn2gp:.4f} vs {ent_center_curv:.4f}, "
        f"outer: {ent_outer_dnn2gp:.4f} vs {ent_outer_curv:.4f}"
    )


if __name__ == "__main__":
    main()
