#!/usr/bin/env python3
"""Heteroscedastic 2D binary classification for curvature-weighted DNN2GP comparison.

Dataset:
- x ~ Uniform([-3,3]^2)
- f(x) = x1 + 0.7 sin(2 x2)
- sigma(x) = 0.15 + 1.5 exp(-(x1^2 + x2^2)/(1.2^2))
- y = sign(f(x) + eps), eps ~ N(0, sigma(x)^2), mapped to {0,1}

Methods compared:
- MAP network
- original DNN2GP approximation
- curvature-weighted extension
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
    parser = argparse.ArgumentParser(description="Heteroscedastic synthetic 2D curvature experiment.")
    parser.add_argument("--output-dir", type=str, default="results/heteroscedastic_curvature")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--n-train", type=int, default=3500)
    parser.add_argument("--n-test", type=int, default=2500)
    parser.add_argument("--laplace-train-size", type=int, default=1400)

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
    parser.add_argument("--kernel-size", type=int, default=240)

    parser.add_argument("--noise-quantile", type=float, default=0.7)
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


def decision_fn(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 0.7 * np.sin(2.0 * x[:, 1])


def sigma_fn(x: np.ndarray) -> np.ndarray:
    return 0.15 + 1.5 * np.exp(-((x[:, 0] ** 2 + x[:, 1] ** 2) / (1.2**2)))


def true_prob_from_gaussian_noise(x: np.ndarray) -> np.ndarray:
    # P(y=1|x) = P(f(x) + eps > 0) = Phi(f/sigma)
    f = decision_fn(x)
    sig = np.clip(sigma_fn(x), a_min=1e-12, a_max=None)
    z = f / sig
    z_t = torch.from_numpy(z).double()
    p = 0.5 * (1.0 + torch.erf(z_t / np.sqrt(2.0)))
    return p.numpy()


def sample_heteroscedastic_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(low=-3.0, high=3.0, size=(n, 2)).astype(np.float64)
    f = decision_fn(x)
    sigma = sigma_fn(x)
    eps = rng.normal(loc=0.0, scale=sigma, size=(n,))
    y_sign = np.sign(f + eps)
    y = (y_sign > 0.0).astype(np.int64)
    p_true = true_prob_from_gaussian_noise(x)
    return x, y, sigma, p_true


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
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


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

    prob_map_list, pred_map_list, ent_map_list = [], [], []
    prob_orig_list, pred_orig_list, ent_orig_list = [], [], []
    prob_curv_list, pred_curv_list, ent_curv_list = [], [], []
    lambda_trace_list = []

    kernel_j, kernel_j_curv, kernel_labels = [], [], []

    iterator = tqdm(range(x_np.shape[0]), desc="Evaluating MAP/DNN2GP/curvature", leave=False)
    for i in iterator:
        x_single = torch.from_numpy(x_np[i : i + 1]).to(device=device, dtype=torch.double)

        logits, jac = jacobian_and_logits(model, x_single)
        p_map = torch.softmax(logits, dim=0)
        ent_map = float((-p_map * torch.log(p_map.clamp_min(1e-12))).sum().item())
        lam = torch.diag(p_map) - torch.outer(p_map, p_map)
        sqrt_lam = matrix_sqrt_psd(lam)

        cov_logits = torch.einsum("kp,p,mp->km", jac, inv_post_prec, jac)
        var_orig = torch.diag(lam @ cov_logits @ lam).clamp_min(1e-12)
        var_curv = torch.diag(sqrt_lam @ cov_logits @ sqrt_lam).clamp_min(1e-12)

        probs_orig, ent_orig = mc_probs_from_diag_logit_var(logits, var_orig, mc_samples, generator)
        probs_curv, ent_curv = mc_probs_from_diag_logit_var(logits, var_curv, mc_samples, generator)

        prob_map_list.append(float(p_map[1].item()))
        pred_map_list.append(int(torch.argmax(p_map).item()))
        ent_map_list.append(ent_map)

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
            kernel_labels.append(-1 if y_np is None else int(y_np[i]))

    out: dict[str, np.ndarray | float] = {
        "prob_map": np.asarray(prob_map_list, dtype=np.float64),
        "pred_map": np.asarray(pred_map_list, dtype=np.int64),
        "entropy_map": np.asarray(ent_map_list, dtype=np.float64),
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


def binary_nll(y_true: np.ndarray, p1: np.ndarray) -> float:
    p = np.clip(p1, a_min=1e-12, a_max=1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, p1: np.ndarray) -> float:
    p = np.clip(p1, a_min=1e-12, a_max=1.0 - 1e-12)
    return float(np.mean((p - y_true) ** 2))


def save_dataset_sigma_map(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    sigma_grid: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(
        sigma_grid,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="magma",
        aspect="equal",
    )
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=10, alpha=0.75, edgecolors="none")
    ax.set_title("Heteroscedastic dataset: training points over noise map sigma(x)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("sigma(x)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_three_maps(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    map3: np.ndarray,
    titles: tuple[str, str, str],
    cmap: str,
    dpi: int,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.9))
    for ax, z, title in zip(axs, [map1, map2, map3], titles):
        im = ax.imshow(
            z,
            origin="lower",
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            cmap=cmap,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.15)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_gp_difference_maps(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    prob_diff: np.ndarray,
    ent_diff: np.ndarray,
    dpi: int,
) -> None:
    vmax_prob = float(np.max(np.abs(prob_diff)))
    vmax_ent = float(np.max(np.abs(ent_diff)))

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.8))

    im0 = axs[0].imshow(
        prob_diff,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="coolwarm",
        vmin=-vmax_prob,
        vmax=vmax_prob,
        aspect="equal",
    )
    axs[0].set_title("Probability diff (curvature - DNN2GP)")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    axs[0].grid(alpha=0.15)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        ent_diff,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="coolwarm",
        vmin=-vmax_ent,
        vmax=vmax_ent,
        aspect="equal",
    )
    axs[1].set_title("Entropy diff (curvature - DNN2GP)")
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    axs[1].grid(alpha=0.15)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

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

    x_train, y_train, sigma_train, p_train_true = sample_heteroscedastic_dataset(args.n_train, rng_train)
    x_test, y_test, sigma_test, p_test_true = sample_heteroscedastic_dataset(args.n_test, rng_test)

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

    laplace_size = min(args.laplace_train_size, x_train.shape[0])
    lap_idx = np.random.default_rng(args.seed + 2).choice(x_train.shape[0], size=laplace_size, replace=False)
    lap_loader = make_loader(x_train[lap_idx], y_train[lap_idx], batch_size=args.laplace_batch_size, shuffle=False)

    model_double = copy.deepcopy(model).to(device=device, dtype=torch.double).eval()
    post_prec = compute_laplace(model=model_double, train_loader=lap_loader, prior_prec=args.prior_prec, device=device)
    inv_post_prec = torch.reciprocal(post_prec.clamp_min(1e-12))

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

    sigma_grid = sigma_fn(x_grid).reshape(args.grid_size, args.grid_size)
    p_true_grid = true_prob_from_gaussian_noise(x_grid).reshape(args.grid_size, args.grid_size)

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

    prob_map_grid = np.asarray(grid_eval["prob_map"]).reshape(args.grid_size, args.grid_size)
    prob_dnn2gp_grid = np.asarray(grid_eval["prob_dnn2gp"]).reshape(args.grid_size, args.grid_size)
    prob_curv_grid = np.asarray(grid_eval["prob_curvature"]).reshape(args.grid_size, args.grid_size)

    ent_map_grid = np.asarray(grid_eval["entropy_map"]).reshape(args.grid_size, args.grid_size)
    ent_dnn2gp_grid = np.asarray(grid_eval["entropy_dnn2gp"]).reshape(args.grid_size, args.grid_size)
    ent_curv_grid = np.asarray(grid_eval["entropy_curvature"]).reshape(args.grid_size, args.grid_size)

    lambda_trace_grid = np.asarray(grid_eval["lambda_trace"]).reshape(args.grid_size, args.grid_size)

    # High/low-noise regions from sigma quantiles on the evaluation grid.
    q = float(np.clip(args.noise_quantile, 0.5, 0.99))
    sigma_low_thr = float(np.quantile(sigma_grid, 1.0 - q))
    sigma_high_thr = float(np.quantile(sigma_grid, q))
    low_noise_mask = sigma_grid <= sigma_low_thr
    high_noise_mask = sigma_grid >= sigma_high_thr

    def region_entropy_stats(ent: np.ndarray) -> tuple[float, float]:
        return float(np.mean(ent[high_noise_mask])), float(np.mean(ent[low_noise_mask]))

    ent_hi_map, ent_lo_map = region_entropy_stats(ent_map_grid)
    ent_hi_dnn2gp, ent_lo_dnn2gp = region_entropy_stats(ent_dnn2gp_grid)
    ent_hi_curv, ent_lo_curv = region_entropy_stats(ent_curv_grid)

    # Test metrics.
    p_map_test = np.asarray(test_eval["prob_map"])
    p_dnn2gp_test = np.asarray(test_eval["prob_dnn2gp"])
    p_curv_test = np.asarray(test_eval["prob_curvature"])

    metrics_map = {
        "accuracy": float(test_eval["accuracy_map"]),
        "nll": binary_nll(y_test, p_map_test),
        "brier": brier_score(y_test, p_map_test),
    }
    metrics_dnn2gp = {
        "accuracy": float(test_eval["accuracy_dnn2gp"]),
        "nll": binary_nll(y_test, p_dnn2gp_test),
        "brier": brier_score(y_test, p_dnn2gp_test),
    }
    metrics_curv = {
        "accuracy": float(test_eval["accuracy_curvature"]),
        "nll": binary_nll(y_test, p_curv_test),
        "brier": brier_score(y_test, p_curv_test),
    }

    # Figures requested.
    fig_dataset = out_dir / "dataset_with_sigma_map.png"
    save_dataset_sigma_map(fig_dataset, xx, yy, sigma_grid, x_train, y_train, args.dpi)

    fig_prob = out_dir / "predictive_probability_maps.png"
    save_three_maps(
        fig_prob,
        xx,
        yy,
        prob_map_grid,
        prob_dnn2gp_grid,
        prob_curv_grid,
        (
            "MAP predictive p(y=1|x)",
            "DNN2GP predictive p(y=1|x)",
            "Curvature predictive p(y=1|x)",
        ),
        cmap="viridis",
        dpi=args.dpi,
    )

    fig_ent = out_dir / "predictive_entropy_maps.png"
    save_three_maps(
        fig_ent,
        xx,
        yy,
        ent_map_grid,
        ent_dnn2gp_grid,
        ent_curv_grid,
        (
            "MAP predictive entropy",
            "DNN2GP predictive entropy",
            "Curvature predictive entropy",
        ),
        cmap="magma",
        dpi=args.dpi,
    )

    fig_diff = out_dir / "gp_difference_maps.png"
    save_gp_difference_maps(
        fig_diff,
        xx,
        yy,
        prob_curv_grid - prob_dnn2gp_grid,
        ent_curv_grid - ent_dnn2gp_grid,
        dpi=args.dpi,
    )

    # Save arrays.
    np.save(out_dir / "kernel_original.npy", np.asarray(test_eval["kernel_original"]))
    np.save(out_dir / "kernel_curvature.npy", np.asarray(test_eval["kernel_curvature"]))
    np.save(out_dir / "kernel_original_normalized.npy", np.asarray(test_eval["kernel_original_normalized"]))
    np.save(out_dir / "kernel_curvature_normalized.npy", np.asarray(test_eval["kernel_curvature_normalized"]))

    np.savez(
        out_dir / "heteroscedastic_curvature_arrays.npz",
        x_train=x_train,
        y_train=y_train,
        sigma_train=sigma_train,
        p_train_true=p_train_true,
        x_test=x_test,
        y_test=y_test,
        sigma_test=sigma_test,
        p_test_true=p_test_true,
        xx=xx,
        yy=yy,
        sigma_grid=sigma_grid,
        true_prob_grid=p_true_grid,
        lambda_trace_grid=lambda_trace_grid,
        prob_map_grid=prob_map_grid,
        prob_dnn2gp_grid=prob_dnn2gp_grid,
        prob_curvature_grid=prob_curv_grid,
        entropy_map_grid=ent_map_grid,
        entropy_dnn2gp_grid=ent_dnn2gp_grid,
        entropy_curvature_grid=ent_curv_grid,
        test_prob_map=p_map_test,
        test_prob_dnn2gp=p_dnn2gp_test,
        test_prob_curvature=p_curv_test,
        test_pred_map=np.asarray(test_eval["pred_map"]),
        test_pred_dnn2gp=np.asarray(test_eval["pred_dnn2gp"]),
        test_pred_curvature=np.asarray(test_eval["pred_curvature"]),
        test_entropy_map=np.asarray(test_eval["entropy_map"]),
        test_entropy_dnn2gp=np.asarray(test_eval["entropy_dnn2gp"]),
        test_entropy_curvature=np.asarray(test_eval["entropy_curvature"]),
        high_noise_mask=high_noise_mask,
        low_noise_mask=low_noise_mask,
    )

    summary = {
        "experiment": "heteroscedastic_curvature_binary",
        "device": str(device),
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "laplace_train_size": int(laplace_size),
        "grid_size": int(args.grid_size),
        "map_train_accuracy": float(train_acc_map),
        "map_test_accuracy": float(test_acc_map),
        "metrics": {
            "map": metrics_map,
            "dnn2gp": metrics_dnn2gp,
            "curvature": metrics_curv,
        },
        "entropy_high_noise_region": {
            "map": ent_hi_map,
            "dnn2gp": ent_hi_dnn2gp,
            "curvature": ent_hi_curv,
        },
        "entropy_low_noise_region": {
            "map": ent_lo_map,
            "dnn2gp": ent_lo_dnn2gp,
            "curvature": ent_lo_curv,
        },
        "noise_region_definition": {
            "sigma_quantile": q,
            "sigma_high_threshold": sigma_high_thr,
            "sigma_low_threshold": sigma_low_thr,
            "high_region_fraction": float(np.mean(high_noise_mask)),
            "low_region_fraction": float(np.mean(low_noise_mask)),
        },
        "kernel_corr_upper_tri_normalized": float(test_eval["kernel_corr"]),
        "lambda_trace_grid_mean": float(np.mean(lambda_trace_grid)),
        "lambda_trace_grid_std": float(np.std(lambda_trace_grid)),
        "sigma_grid_mean": float(np.mean(sigma_grid)),
        "sigma_grid_std": float(np.std(sigma_grid)),
        "figures": {
            "dataset_sigma": str(fig_dataset),
            "probability_maps": str(fig_prob),
            "entropy_maps": str(fig_ent),
            "gp_difference_maps": str(fig_diff),
        },
    }

    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs in: {out_dir}")
    print(f"Saved summary: {summary_path}")
    print(
        "Test accuracy | MAP: "
        f"{metrics_map['accuracy']:.4f}, DNN2GP: {metrics_dnn2gp['accuracy']:.4f}, Curvature: {metrics_curv['accuracy']:.4f}"
    )
    print(
        "Test NLL | MAP: "
        f"{metrics_map['nll']:.4f}, DNN2GP: {metrics_dnn2gp['nll']:.4f}, Curvature: {metrics_curv['nll']:.4f}"
    )


if __name__ == "__main__":
    main()
