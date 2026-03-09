#!/usr/bin/env python3
"""Hard-band heteroscedastic 2D experiment for NTK vs curvature-weighted DNN2GP.

Design goal:
- Create a regime with strongly heterogeneous ambiguity localized near a curved
  decision band, so curvature weighting can change uncertainty structure.
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
    parser = argparse.ArgumentParser(description="Curvature hard-band 2D experiment.")
    parser.add_argument("--output-dir", type=str, default="results/curvature_hardband_2d")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--n-train", type=int, default=3200)
    parser.add_argument("--n-test", type=int, default=2800)
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

    parser.add_argument("--kernel-size", type=int, default=260)
    parser.add_argument("--hard-quantile", type=float, default=0.78)
    parser.add_argument("--train-hard-fraction", type=float, default=0.20)
    parser.add_argument("--train-pool-mult", type=int, default=10)

    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_neurips_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 120,
            "savefig.dpi": 220,
        }
    )


def resolve_device(device_arg: str) -> torch.device:
    req = device_arg.lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def phi(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 0.7 * np.sin(2.0 * x[:, 1])


def sigma_fn(x: np.ndarray) -> np.ndarray:
    # High noise near the curved boundary (phi ~ 0), plus central bump.
    return 0.08 + 2.2 * np.exp(-(phi(x) ** 2) / (0.18**2)) + 0.6 * np.exp(-((x[:, 0] ** 2 + x[:, 1] ** 2) / (1.0**2)))


def true_prob_fn(x: np.ndarray) -> np.ndarray:
    # p(y=1|x) = P(phi + eps > 0), eps ~ N(0, sigma^2)
    z = phi(x) / np.clip(sigma_fn(x), a_min=1e-12, a_max=None)
    z_t = torch.from_numpy(z).double()
    return (0.5 * (1.0 + torch.erf(z_t / np.sqrt(2.0)))).numpy()


def sample_raw(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(-3.0, 3.0, size=(n, 2)).astype(np.float64)
    sigma = sigma_fn(x)
    eps = rng.normal(0.0, sigma, size=(n,))
    y = (phi(x) + eps > 0.0).astype(np.int64)
    p_true = true_prob_fn(x)
    return x, y, sigma, p_true


def build_train_set(
    n_train: int,
    hard_quantile: float,
    hard_fraction: float,
    pool_mult: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pool_n = max(n_train * pool_mult, 8000)
    x_pool, y_pool, sigma_pool, p_true_pool = sample_raw(pool_n, rng)

    thr = float(np.quantile(sigma_pool, hard_quantile))
    hard_idx = np.where(sigma_pool >= thr)[0]
    easy_idx = np.where(sigma_pool < thr)[0]

    n_hard = int(round(n_train * hard_fraction))
    n_easy = n_train - n_hard

    if len(hard_idx) < n_hard or len(easy_idx) < n_easy:
        raise RuntimeError("Not enough hard/easy samples in pool. Increase --train-pool-mult.")

    sel_hard = rng.choice(hard_idx, size=n_hard, replace=False)
    sel_easy = rng.choice(easy_idx, size=n_easy, replace=False)
    sel = np.concatenate([sel_hard, sel_easy])
    rng.shuffle(sel)

    return x_pool[sel], y_pool[sel], sigma_pool[sel], p_true_pool[sel]


class BinaryMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, n_hidden: int = 2, activation: str = "tanh"):
        super().__init__()
        self.fc_in = nn.Linear(2, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(max(0, n_hidden - 1))])
        self.fc_out = nn.Linear(hidden_size, 2)
        self.activation_name = activation

    def transfer(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(x)
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transfer(self.fc_in(x))
        for layer in self.hidden_layers:
            x = self.transfer(layer(x))
        return self.fc_out(x)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def map_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_total, n_correct = 0, 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)
        pred = model(xb).argmax(dim=1)
        n_correct += int((pred == yb).sum().item())
        n_total += int(yb.numel())
    return n_correct / max(1, n_total)


def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n_items = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            bs = int(xb.shape[0])
            loss_sum += float(loss.item()) * bs
            n_items += bs

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            tr_acc = map_accuracy(model, train_loader, device)
            te_acc = map_accuracy(model, test_loader, device)
            print(f"epoch {ep:03d} | loss {loss_sum / max(1, n_items):.4f} | train acc {tr_acc:.4f} | test acc {te_acc:.4f}")


def gradient_vector(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).flatten())
        else:
            chunks.append(p.grad.detach().flatten())
    return torch.cat(chunks, dim=0)


def jacobian_and_logits(model: nn.Module, x_one: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(x_one).squeeze(0)
    rows = []
    for c in range(logits.numel()):
        retain = c < logits.numel() - 1
        logits[c].backward(retain_graph=retain)
        rows.append(gradient_vector(model))
        model.zero_grad(set_to_none=True)
    return logits.detach(), torch.stack(rows, dim=0).detach()


def matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    m = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(m)
    evals = evals.clamp_min(eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def mc_probs(mean_logits: torch.Tensor, var_diag: torch.Tensor, mc_samples: int, generator: torch.Generator) -> tuple[torch.Tensor, float]:
    eps = torch.randn(
        (mc_samples, mean_logits.numel()),
        dtype=mean_logits.dtype,
        device=mean_logits.device,
        generator=generator,
    )
    logits_mc = mean_logits.unsqueeze(0) + eps * torch.sqrt(var_diag.clamp_min(1e-12)).unsqueeze(0)
    probs = torch.softmax(logits_mc, dim=1)
    p_mean = probs.mean(dim=0)
    ent = float((-p_mean * torch.log(p_mean.clamp_min(1e-12))).sum().item())
    return p_mean, ent


def normalize_kernel(k: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(k), a_min=1e-12, a_max=None))
    return k / (d[:, None] * d[None, :])


def eval_methods(
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
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()

    out = {
        "p_map": [],
        "p_ntk": [],
        "p_curv": [],
        "pred_map": [],
        "pred_ntk": [],
        "pred_curv": [],
        "ent_map": [],
        "ent_ntk": [],
        "ent_curv": [],
        "lambda_trace": [],
    }

    kJ, kJc, klabels = [], [], []

    for i in tqdm(range(x_np.shape[0]), desc="Evaluating MAP/NTK/curvature", leave=False):
        x_one = torch.from_numpy(x_np[i : i + 1]).to(device=device, dtype=torch.double)
        logits, jac = jacobian_and_logits(model, x_one)

        p_map = torch.softmax(logits, dim=0)
        ent_map = float((-p_map * torch.log(p_map.clamp_min(1e-12))).sum().item())

        lam = torch.diag(p_map) - torch.outer(p_map, p_map)
        sqrt_lam = matrix_sqrt_psd(lam)

        cov_logits = torch.einsum("kp,p,mp->km", jac, inv_post_prec, jac)
        # NTK baseline: no Lambda weighting.
        var_ntk = torch.diag(cov_logits).clamp_min(1e-12)
        # Curvature extension: sqrt(Lambda) weighting.
        var_curv = torch.diag(sqrt_lam @ cov_logits @ sqrt_lam).clamp_min(1e-12)

        p_ntk, ent_ntk = mc_probs(logits, var_ntk, mc_samples, generator)
        p_curv, ent_curv = mc_probs(logits, var_curv, mc_samples, generator)

        out["p_map"].append(float(p_map[1].item()))
        out["p_ntk"].append(float(p_ntk[1].item()))
        out["p_curv"].append(float(p_curv[1].item()))

        out["pred_map"].append(int(torch.argmax(p_map).item()))
        out["pred_ntk"].append(int(torch.argmax(p_ntk).item()))
        out["pred_curv"].append(int(torch.argmax(p_curv).item()))

        out["ent_map"].append(ent_map)
        out["ent_ntk"].append(ent_ntk)
        out["ent_curv"].append(ent_curv)
        out["lambda_trace"].append(float(torch.trace(lam).item()))

        if collect_kernel and len(kJ) < kernel_size:
            kJ.append(jac.detach().cpu().float().numpy())
            kJc.append((sqrt_lam @ jac).detach().cpu().float().numpy())
            klabels.append(-1 if y_np is None else int(y_np[i]))

    for k in list(out.keys()):
        out[k] = np.asarray(out[k])

    if y_np is not None:
        out["acc_map"] = float(np.mean(out["pred_map"] == y_np))
        out["acc_ntk"] = float(np.mean(out["pred_ntk"] == y_np))
        out["acc_curv"] = float(np.mean(out["pred_curv"] == y_np))

    if collect_kernel and len(kJ) > 1:
        J = np.stack(kJ, axis=0)
        Jc = np.stack(kJc, axis=0)
        labels = np.asarray(klabels)

        # Group by (label, hard/easy surrogate via label blocks only here).
        sort_idx = np.argsort(labels, kind="stable")
        J = J[sort_idx]
        Jc = Jc[sort_idx]
        labels = labels[sort_idx]

        K_ntk = np.einsum("ikp,jkp->ij", J, J, optimize=True)
        K_curv = np.einsum("ikp,jkp->ij", Jc, Jc, optimize=True)
        K_ntk_n = normalize_kernel(K_ntk)
        K_curv_n = normalize_kernel(K_curv)

        tri = np.triu_indices_from(K_ntk_n, k=1)
        corr = float(np.corrcoef(K_ntk_n[tri], K_curv_n[tri])[0, 1]) if tri[0].size > 1 else float("nan")

        out["K_ntk"] = K_ntk
        out["K_curv"] = K_curv
        out["K_ntk_n"] = K_ntk_n
        out["K_curv_n"] = K_curv_n
        out["K_labels"] = labels
        out["kernel_corr"] = corr

    return out


def binary_nll(y_true: np.ndarray, p1: np.ndarray) -> float:
    p = np.clip(p1, a_min=1e-12, a_max=1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier(y_true: np.ndarray, p1: np.ndarray) -> float:
    p = np.clip(p1, a_min=1e-12, a_max=1.0 - 1e-12)
    return float(np.mean((p - y_true) ** 2))


def ece_binary(y_true: np.ndarray, p1: np.ndarray, n_bins: int = 15) -> float:
    p = np.clip(p1, a_min=1e-12, a_max=1 - 1e-12)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if np.any(m):
            ece += np.abs(np.mean(correct[m]) - np.mean(conf[m])) * np.mean(m)
    return float(ece)


def risk_coverage(y_true: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, coverages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(uncertainty)  # least uncertain first
    y_sorted = y_true[idx]
    p_sorted = pred[idx]

    risks = []
    for c in coverages:
        k = max(1, int(np.ceil(c * len(y_true))))
        acc = np.mean(p_sorted[:k] == y_sorted[:k])
        risks.append(1.0 - acc)
    return coverages, np.asarray(risks)


def plot_dataset(out_path: Path, xx: np.ndarray, yy: np.ndarray, sigma_grid: np.ndarray, p_true_grid: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, hard_thr: float, dpi: int) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.7))

    im0 = axs[0].imshow(
        sigma_grid,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="magma",
        aspect="equal",
    )
    axs[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=8, alpha=0.62, edgecolors="none")
    axs[0].contour(xx, yy, sigma_grid, levels=[hard_thr], colors="white", linewidths=1.0, linestyles="--")
    axs[0].set_title("Train set on noise map $\\sigma(x)$")
    axs[0].set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        p_true_grid,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axs[1].contour(xx, yy, p_true_grid, levels=[0.5], colors="white", linewidths=1.2)
    axs[1].set_title("True probability $p(y{=}1\\mid x)$")
    axs[1].set_xlabel("$x_1$")
    axs[1].set_ylabel("$x_2$")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_prob_maps(out_path: Path, xx: np.ndarray, yy: np.ndarray, p_map: np.ndarray, p_ntk: np.ndarray, p_curv: np.ndarray, dpi: int) -> None:
    diff = p_curv - p_ntk
    vmax_diff = float(np.max(np.abs(diff)))

    fig, axs = plt.subplots(1, 4, figsize=(15.8, 4.2))
    titles = ["MAP", "DNN2GP-NTK", "Curvature-weighted", "Curvature - NTK"]
    maps = [p_map, p_ntk, p_curv, diff]
    cmaps = ["viridis", "viridis", "viridis", "coolwarm"]
    vmins = [0.0, 0.0, 0.0, -vmax_diff]
    vmaxs = [1.0, 1.0, 1.0, vmax_diff]

    for ax, z, t, c, v0, v1 in zip(axs, maps, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(z, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=c, vmin=v0, vmax=v1, aspect="equal")
        ax.contour(xx, yy, z, levels=[0.5], colors="white", linewidths=1.0)
        ax.set_title(t)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_maps(out_path: Path, xx: np.ndarray, yy: np.ndarray, ent_map: np.ndarray, ent_ntk: np.ndarray, ent_curv: np.ndarray, dpi: int) -> None:
    diff = ent_curv - ent_ntk
    vmax_diff = float(np.max(np.abs(diff)))

    fig, axs = plt.subplots(1, 4, figsize=(15.8, 4.2))
    titles = ["MAP entropy", "NTK entropy", "Curvature entropy", "Curvature - NTK"]
    maps = [ent_map, ent_ntk, ent_curv, diff]
    cmaps = ["magma", "magma", "magma", "coolwarm"]
    vmins = [0.0, 0.0, 0.0, -vmax_diff]
    vmaxs = [float(np.max(ent_map)), float(np.max(ent_ntk)), float(np.max(ent_curv)), vmax_diff]

    for ax, z, t, c, v0, v1 in zip(axs, maps, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(z, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=c, vmin=v0, vmax=v1, aspect="equal")
        ax.set_title(t)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_maps(out_path: Path, K_ntk_n: np.ndarray, K_curv_n: np.ndarray, labels: np.ndarray, dpi: int) -> None:
    diff = K_curv_n - K_ntk_n
    vmax_diff = float(np.max(np.abs(diff)))

    fig, axs = plt.subplots(1, 3, figsize=(12.3, 4.1))
    ims = [
        axs[0].imshow(K_ntk_n, cmap="magma", aspect="auto"),
        axs[1].imshow(K_curv_n, cmap="magma", aspect="auto"),
        axs[2].imshow(diff, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff, aspect="auto"),
    ]
    axs[0].set_title("Normalized NTK kernel")
    axs[1].set_title("Normalized curvature kernel")
    axs[2].set_title("Kernel diff (curv - NTK)")
    for ax in axs:
        ax.set_xlabel("sample index")
        ax.set_ylabel("sample index")
        for b in np.where(np.diff(labels) != 0)[0]:
            ax.axhline(b + 0.5, color="white", lw=0.4, alpha=0.7)
            ax.axvline(b + 0.5, color="white", lw=0.4, alpha=0.7)

    for ax, im in zip(axs, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(out_path: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    palette = {
        "MAP": "#1f77b4",
        "NTK": "#2ca02c",
        "Curvature": "#d62728",
    }
    for name, (cov, risk) in curves.items():
        ax.plot(cov * 100.0, risk * 100.0, marker="o", ms=4, label=name, color=palette.get(name, None))

    ax.set_xlabel("Coverage (%) on hard region")
    ax.set_ylabel("Risk = 1 - accuracy (%)")
    ax.set_title("Selective classification on hard region")
    ax.set_xlim(15, 101)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_neurips_style()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    rng_train = np.random.default_rng(args.seed)
    rng_test = np.random.default_rng(args.seed + 1)

    x_train, y_train, sigma_train, p_train_true = build_train_set(
        n_train=args.n_train,
        hard_quantile=args.hard_quantile,
        hard_fraction=args.train_hard_fraction,
        pool_mult=args.train_pool_mult,
        rng=rng_train,
    )
    x_test, y_test, sigma_test, p_test_true = sample_raw(args.n_test, rng_test)

    train_loader = make_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size=512, shuffle=False)

    model = BinaryMLP(hidden_size=args.hidden_size, n_hidden=args.n_hidden, activation=args.activation).to(
        device=device, dtype=torch.float32
    )
    train_map(model, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay)

    tr_acc_map = map_accuracy(model, make_loader(x_train, y_train, batch_size=512, shuffle=False), device)
    te_acc_map = map_accuracy(model, test_loader, device)

    lap_size = min(args.laplace_train_size, len(x_train))
    lap_idx = np.random.default_rng(args.seed + 2).choice(len(x_train), size=lap_size, replace=False)
    lap_loader = make_loader(x_train[lap_idx], y_train[lap_idx], batch_size=args.laplace_batch_size, shuffle=False)

    model_d = copy.deepcopy(model).to(device=device, dtype=torch.double).eval()
    post_prec = compute_laplace(model=model_d, train_loader=lap_loader, prior_prec=args.prior_prec, device=device)
    inv_post_prec = torch.reciprocal(post_prec.clamp_min(1e-12))

    test_eval = eval_methods(
        model=model_d,
        inv_post_prec=inv_post_prec,
        x_np=x_test,
        y_np=y_test,
        mc_samples=args.mc_samples,
        seed=args.seed + 50,
        kernel_size=args.kernel_size,
        collect_kernel=True,
    )

    grid_lin = np.linspace(args.grid_min, args.grid_max, args.grid_size)
    xx, yy = np.meshgrid(grid_lin, grid_lin)
    x_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    sigma_grid = sigma_fn(x_grid).reshape(args.grid_size, args.grid_size)
    p_true_grid = true_prob_fn(x_grid).reshape(args.grid_size, args.grid_size)

    grid_eval = eval_methods(
        model=model_d,
        inv_post_prec=inv_post_prec,
        x_np=x_grid,
        y_np=None,
        mc_samples=args.mc_samples,
        seed=args.seed + 70,
        kernel_size=0,
        collect_kernel=False,
    )

    p_map_grid = grid_eval["p_map"].reshape(args.grid_size, args.grid_size)
    p_ntk_grid = grid_eval["p_ntk"].reshape(args.grid_size, args.grid_size)
    p_curv_grid = grid_eval["p_curv"].reshape(args.grid_size, args.grid_size)

    ent_map_grid = grid_eval["ent_map"].reshape(args.grid_size, args.grid_size)
    ent_ntk_grid = grid_eval["ent_ntk"].reshape(args.grid_size, args.grid_size)
    ent_curv_grid = grid_eval["ent_curv"].reshape(args.grid_size, args.grid_size)
    lambda_trace_grid = grid_eval["lambda_trace"].reshape(args.grid_size, args.grid_size)

    # Hard / low noise regions for region-wise metrics.
    high_thr = float(np.quantile(sigma_test, args.hard_quantile))
    low_thr = float(np.quantile(sigma_test, 1.0 - args.hard_quantile))
    hard_test_mask = sigma_test >= high_thr
    low_test_mask = sigma_test <= low_thr

    high_thr_grid = float(np.quantile(sigma_grid, args.hard_quantile))

    # Metrics.
    def metric_block(y: np.ndarray, p: np.ndarray, bins: int) -> dict[str, float]:
        pred = (p >= 0.5).astype(np.int64)
        return {
            "accuracy": float(np.mean(pred == y)),
            "nll": binary_nll(y, p),
            "brier": brier(y, p),
            "ece": ece_binary(y, p, n_bins=bins),
        }

    metrics_map = metric_block(y_test, test_eval["p_map"], args.ece_bins)
    metrics_ntk = metric_block(y_test, test_eval["p_ntk"], args.ece_bins)
    metrics_curv = metric_block(y_test, test_eval["p_curv"], args.ece_bins)

    def region_stats(ent: np.ndarray, p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        return {
            "entropy_mean": float(np.mean(ent[mask])),
            "nll": binary_nll(y[mask], p[mask]),
            "brier": brier(y[mask], p[mask]),
            "count": int(np.sum(mask)),
        }

    hard_stats = {
        "map": region_stats(test_eval["ent_map"], test_eval["p_map"], y_test, hard_test_mask),
        "ntk": region_stats(test_eval["ent_ntk"], test_eval["p_ntk"], y_test, hard_test_mask),
        "curvature": region_stats(test_eval["ent_curv"], test_eval["p_curv"], y_test, hard_test_mask),
    }
    low_stats = {
        "map": region_stats(test_eval["ent_map"], test_eval["p_map"], y_test, low_test_mask),
        "ntk": region_stats(test_eval["ent_ntk"], test_eval["p_ntk"], y_test, low_test_mask),
        "curvature": region_stats(test_eval["ent_curv"], test_eval["p_curv"], y_test, low_test_mask),
    }

    # Risk-coverage on hard region.
    cov = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    curves = {
        "MAP": risk_coverage(y_test[hard_test_mask], test_eval["pred_map"][hard_test_mask], test_eval["ent_map"][hard_test_mask], cov),
        "NTK": risk_coverage(y_test[hard_test_mask], test_eval["pred_ntk"][hard_test_mask], test_eval["ent_ntk"][hard_test_mask], cov),
        "Curvature": risk_coverage(
            y_test[hard_test_mask],
            test_eval["pred_curv"][hard_test_mask],
            test_eval["ent_curv"][hard_test_mask],
            cov,
        ),
    }

    # Figures.
    fig_dataset = out_dir / "dataset_and_noise_maps_neurips.png"
    plot_dataset(fig_dataset, xx, yy, sigma_grid, p_true_grid, x_train, y_train, high_thr_grid, args.dpi)

    fig_prob = out_dir / "predictive_probability_maps_neurips.png"
    plot_prob_maps(fig_prob, xx, yy, p_map_grid, p_ntk_grid, p_curv_grid, args.dpi)

    fig_ent = out_dir / "predictive_entropy_maps_neurips.png"
    plot_entropy_maps(fig_ent, xx, yy, ent_map_grid, ent_ntk_grid, ent_curv_grid, args.dpi)

    fig_kernel = out_dir / "kernel_comparison_neurips.png"
    plot_kernel_maps(fig_kernel, test_eval["K_ntk_n"], test_eval["K_curv_n"], test_eval["K_labels"], args.dpi)

    fig_rc = out_dir / "hard_region_risk_coverage_neurips.png"
    plot_risk_coverage(fig_rc, curves, args.dpi)

    # Save arrays and kernels.
    np.save(out_dir / "kernel_ntk.npy", test_eval["K_ntk"])
    np.save(out_dir / "kernel_curvature.npy", test_eval["K_curv"])
    np.save(out_dir / "kernel_ntk_normalized.npy", test_eval["K_ntk_n"])
    np.save(out_dir / "kernel_curvature_normalized.npy", test_eval["K_curv_n"])

    np.savez(
        out_dir / "curvature_hardband_arrays.npz",
        x_train=x_train,
        y_train=y_train,
        sigma_train=sigma_train,
        p_train_true=p_train_true,
        x_test=x_test,
        y_test=y_test,
        sigma_test=sigma_test,
        p_test_true=p_test_true,
        hard_test_mask=hard_test_mask,
        low_test_mask=low_test_mask,
        xx=xx,
        yy=yy,
        sigma_grid=sigma_grid,
        p_true_grid=p_true_grid,
        p_map_grid=p_map_grid,
        p_ntk_grid=p_ntk_grid,
        p_curv_grid=p_curv_grid,
        ent_map_grid=ent_map_grid,
        ent_ntk_grid=ent_ntk_grid,
        ent_curv_grid=ent_curv_grid,
        lambda_trace_grid=lambda_trace_grid,
        p_map_test=test_eval["p_map"],
        p_ntk_test=test_eval["p_ntk"],
        p_curv_test=test_eval["p_curv"],
        pred_map_test=test_eval["pred_map"],
        pred_ntk_test=test_eval["pred_ntk"],
        pred_curv_test=test_eval["pred_curv"],
        ent_map_test=test_eval["ent_map"],
        ent_ntk_test=test_eval["ent_ntk"],
        ent_curv_test=test_eval["ent_curv"],
    )

    summary = {
        "experiment": "curvature_hardband_2d",
        "device": str(device),
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "laplace_train_size": int(lap_size),
        "train_hard_fraction_target": float(args.train_hard_fraction),
        "train_hard_fraction_realized": float(np.mean(sigma_train >= np.quantile(sigma_train, args.hard_quantile))),
        "map_train_accuracy": float(tr_acc_map),
        "map_test_accuracy": float(te_acc_map),
        "metrics_test": {
            "map": metrics_map,
            "ntk": metrics_ntk,
            "curvature": metrics_curv,
        },
        "high_noise_region_test": hard_stats,
        "low_noise_region_test": low_stats,
        "noise_thresholds": {
            "hard_quantile": float(args.hard_quantile),
            "sigma_high_threshold_test": high_thr,
            "sigma_low_threshold_test": low_thr,
            "hard_region_fraction_test": float(np.mean(hard_test_mask)),
            "low_region_fraction_test": float(np.mean(low_test_mask)),
        },
        "kernel_corr_upper_tri_normalized": float(test_eval["kernel_corr"]),
        "lambda_trace_grid_mean": float(np.mean(lambda_trace_grid)),
        "lambda_trace_grid_std": float(np.std(lambda_trace_grid)),
        "figures": {
            "dataset": str(fig_dataset),
            "probability_maps": str(fig_prob),
            "entropy_maps": str(fig_ent),
            "kernel_maps": str(fig_kernel),
            "risk_coverage": str(fig_rc),
        },
    }

    # Quick verdict.
    if metrics_curv["nll"] + 1e-6 < metrics_ntk["nll"] and hard_stats["curvature"]["nll"] < hard_stats["ntk"]["nll"]:
        summary["quick_take"] = "curvature better calibrated globally and in hard region"
    elif hard_stats["curvature"]["nll"] < hard_stats["ntk"]["nll"]:
        summary["quick_take"] = "curvature improves hard-region calibration but not global metrics"
    else:
        summary["quick_take"] = "curvature does not beat NTK in this run"

    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs in: {out_dir}")
    print(f"Saved summary: {summary_path}")
    print(
        "Test metrics | "
        f"MAP acc/nll: {metrics_map['accuracy']:.4f}/{metrics_map['nll']:.4f}, "
        f"NTK acc/nll: {metrics_ntk['accuracy']:.4f}/{metrics_ntk['nll']:.4f}, "
        f"Curv acc/nll: {metrics_curv['accuracy']:.4f}/{metrics_curv['nll']:.4f}"
    )
    print(
        "Hard-region entropy mean | "
        f"MAP {hard_stats['map']['entropy_mean']:.4f}, "
        f"NTK {hard_stats['ntk']['entropy_mean']:.4f}, "
        f"Curv {hard_stats['curvature']['entropy_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
