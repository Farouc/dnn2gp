#!/usr/bin/env python3
"""Delta tuning on real regression data (Laplace + optional VI).

This script targets real-data validation of the paper's key claim:
test MSE minimum should track train marginal-likelihood minimum.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dual_models import DualGPRegression
from src.neural_networks import SimpleMLP
from src.variational_models import VariationalNeuralRegression


@dataclass
class Config:
    name: str
    dataset: str
    dataset_name: str
    test_size: float
    n_retries: int
    n_params: int
    delta_log10_min: float
    delta_log10_max: float
    n_epochs: int
    vi_epochs: int
    include_vi: bool
    hidden_size: int
    n_layers: int
    activation: str
    sigma_noise: float
    learning_rate: float
    lr_factor: float
    seed: int
    device: str
    dtype: str
    print_every: int
    output_dir: Path
    figure_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Delta sweep on real data: MAP/Laplace vs VI.")
    parser.add_argument("--name", type=str, default="real_delta")
    parser.add_argument("--dataset", type=str, default="diabetes", choices=["diabetes"])
    parser.add_argument("--dataset_name", type=str, default="Diabetes (real regression)")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--n_retries", type=int, default=5)
    parser.add_argument("--n_params", type=int, default=13)
    parser.add_argument("--delta_log10_min", type=float, default=-2.0)
    parser.add_argument("--delta_log10_max", type=float, default=2.0)
    parser.add_argument("--n_epochs", type=int, default=1500)
    parser.add_argument("--vi_epochs", type=int, default=1000)
    parser.add_argument("--include_vi", dest="include_vi", action="store_true")
    parser.add_argument("--no_vi", dest="include_vi", action="store_false")
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "sigmoid", "elu"])
    parser.add_argument("--sigma_noise", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-2)
    parser.add_argument("--lr_factor", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/marglik_delta_real")
    parser.add_argument("--figure_dir", type=str, default="figures")
    parser.set_defaults(include_vi=True)
    args = parser.parse_args()

    return Config(
        name=args.name,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        n_retries=args.n_retries,
        n_params=args.n_params,
        delta_log10_min=args.delta_log10_min,
        delta_log10_max=args.delta_log10_max,
        n_epochs=args.n_epochs,
        vi_epochs=args.vi_epochs,
        include_vi=args.include_vi,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        activation=args.activation,
        sigma_noise=args.sigma_noise,
        learning_rate=args.learning_rate,
        lr_factor=args.lr_factor,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        print_every=args.print_every,
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
    )


def slugify(text: str) -> str:
    txt = text.strip().lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = txt.strip("_")
    return txt or "dataset"


def select_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def select_dtype(dtype_name: str) -> torch.dtype:
    return torch.float64 if dtype_name == "float64" else torch.float32


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
        return X.astype(np.float64), y.astype(np.float64)
    raise ValueError(f"Unsupported dataset: {name}")


def make_splits(cfg: Config) -> List[Dict[str, np.ndarray | float | int]]:
    X_all, y_all = load_base_dataset(cfg.dataset)
    splits: List[Dict[str, np.ndarray | float | int]] = []
    for i in range(cfg.n_retries):
        split_seed = cfg.seed + i
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=cfg.test_size,
            random_state=split_seed,
        )

        x_scaler = StandardScaler(copy=False)
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

        y_train_2d = y_train.reshape(-1, 1)
        y_test_2d = y_test.reshape(-1, 1)
        y_scaler = StandardScaler(copy=False)
        y_train_scaled = y_scaler.fit_transform(y_train_2d).squeeze()
        y_test_scaled = y_scaler.transform(y_test_2d).squeeze()

        splits.append(
            {
                "split_seed": int(split_seed),
                "X_train": X_train.astype(np.float64, copy=False),
                "y_train": y_train_scaled.astype(np.float64, copy=False),
                "X_test": X_test.astype(np.float64, copy=False),
                "y_test": y_test_scaled.astype(np.float64, copy=False),
                "y_scale": float(y_scaler.scale_[0]),
            }
        )
    return splits


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in model.parameters()])


def forward_numpy(model: torch.nn.Module, X: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X, device=device, dtype=dtype)
        pred = model(Xt).squeeze(-1).detach().cpu().numpy()
    return pred.astype(np.float64)


def compute_jacobian_features(model: torch.nn.Module, X: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(X, device=device, dtype=dtype)
    n = Xt.shape[0]
    d = sum(p.numel() for p in model.parameters())
    J = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        model.zero_grad(set_to_none=True)
        yi = model(Xt[i : i + 1]).squeeze()
        yi.backward()
        grad_i = torch.cat([p.grad.view(-1) for p in model.parameters()])
        J[i] = grad_i.detach().cpu().numpy().astype(np.float64, copy=False)
    return J


def train_map_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    delta: float,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.nn.Module, List[float]]:
    torch.manual_seed(seed)
    model = SimpleMLP(
        input_size=X_train.shape[1],
        h_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        activation=cfg.activation,
    ).to(device=device, dtype=dtype)

    X_t = torch.as_tensor(X_train, device=device, dtype=dtype)
    y_t = torch.as_tensor(y_train, device=device, dtype=dtype)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor, min_lr=1e-6, patience=50)
    beta_noise = 1.0 / (cfg.sigma_noise**2)

    losses: List[float] = []
    model.train()
    for epoch in range(cfg.n_epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(X_t).squeeze(-1)
        nll = 0.5 * beta_noise * torch.sum((pred - y_t) ** 2)
        prior_penalty = 0.5 * delta * torch.sum(flatten_parameters(model) ** 2)
        loss = nll + prior_penalty
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach().item())
        losses.append(float(loss.detach().item()))
        if cfg.print_every > 0 and (epoch + 1) % cfg.print_every == 0:
            print(f"    epoch {epoch + 1:5d} | loss={losses[-1]:.4f}")
    return model, losses


def evaluate_vi_branch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_scale: float,
    delta: float,
    cfg: Config,
    seed: int,
) -> Dict[str, float]:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        vi_nn = VariationalNeuralRegression(
            X_train,
            y_train,
            delta,
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_layers,
            n_epochs=cfg.vi_epochs,
            seed=seed,
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    vi_pred_train = vi_nn.posterior_predictive_f(X_train, compute_std=False)
    vi_pred_test = vi_nn.posterior_predictive_f(X_test, compute_std=False)
    train_mse_vi = float(y_scale**2 * np.mean((vi_pred_train - y_train) ** 2))
    test_mse_vi = float(y_scale**2 * np.mean((vi_pred_test - y_test) ** 2))
    log_marglik_vi = float(vi_nn.compute_log_mlh_converged())
    return {
        "train_mse_vi": train_mse_vi,
        "test_mse_vi": test_mse_vi,
        "train_log_marglik_vi": log_marglik_vi,
        "final_train_loss_vi": float(vi_nn.loss),
    }


def evaluate_delta_on_split(
    delta: float,
    split_data: Dict[str, np.ndarray | float | int],
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    X_train = split_data["X_train"]  # type: ignore[assignment]
    y_train = split_data["y_train"]  # type: ignore[assignment]
    X_test = split_data["X_test"]  # type: ignore[assignment]
    y_test = split_data["y_test"]  # type: ignore[assignment]
    y_scale = float(split_data["y_scale"])
    split_seed = int(split_data["split_seed"])

    model, losses = train_map_model(
        X_train=X_train,
        y_train=y_train,
        delta=delta,
        cfg=cfg,
        device=device,
        dtype=dtype,
        seed=split_seed,
    )

    pred_train = forward_numpy(model, X_train, device, dtype)
    pred_test = forward_numpy(model, X_test, device, dtype)
    train_mse = float(y_scale**2 * np.mean((pred_train - y_train) ** 2))
    test_mse = float(y_scale**2 * np.mean((pred_test - y_test) ** 2))

    U_train = compute_jacobian_features(model, X_train, device, dtype)
    theta_star = flatten_parameters(model).detach().cpu().numpy().astype(np.float64, copy=False)
    beta_noise = 1.0 / (cfg.sigma_noise**2)
    vs = beta_noise * (pred_train - y_train)
    y_hat = U_train @ theta_star - vs / beta_noise
    m0 = np.zeros_like(theta_star, dtype=np.float64)
    S0 = (1.0 / delta) * np.eye(theta_star.size, dtype=np.float64)
    gp = DualGPRegression(
        X_hat=U_train,
        y_hat=y_hat,
        s_noise=float(cfg.sigma_noise),
        m_0=m0,
        S_0=S0,
        comp_post=True,
    )
    log_marglik = float(gp.log_marginal_likelihood())

    metrics = {
        "split_seed": split_seed,
        "train_mse_map": train_mse,
        "test_mse_map": test_mse,
        "train_log_marglik": log_marglik,
        "final_train_loss": float(losses[-1]),
    }
    if cfg.include_vi:
        metrics.update(
            evaluate_vi_branch(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                y_scale=y_scale,
                delta=delta,
                cfg=cfg,
                seed=split_seed,
            )
        )
    return metrics


def aggregate(raw_results: List[Dict], include_vi: bool) -> Dict[str, List[float]]:
    delta_values: List[float] = []
    train_mse_mean: List[float] = []
    train_mse_sem: List[float] = []
    test_mse_mean: List[float] = []
    test_mse_sem: List[float] = []
    neg_mlh_mean: List[float] = []
    neg_mlh_sem: List[float] = []
    train_mse_vi_mean: List[float] = []
    train_mse_vi_sem: List[float] = []
    test_mse_vi_mean: List[float] = []
    test_mse_vi_sem: List[float] = []
    neg_mlh_vi_mean: List[float] = []
    neg_mlh_vi_sem: List[float] = []

    for item in raw_results:
        delta_values.append(item["delta"])
        train_mse = np.array([r["train_mse_map"] for r in item["retries"]], dtype=np.float64)
        test_mse = np.array([r["test_mse_map"] for r in item["retries"]], dtype=np.float64)
        neg_mlh = -np.array([r["train_log_marglik"] for r in item["retries"]], dtype=np.float64)
        n = max(1, train_mse.size)
        denom = np.sqrt(n)
        train_mse_mean.append(float(train_mse.mean()))
        train_mse_sem.append(float(train_mse.std(ddof=0) / denom))
        test_mse_mean.append(float(test_mse.mean()))
        test_mse_sem.append(float(test_mse.std(ddof=0) / denom))
        neg_mlh_mean.append(float(neg_mlh.mean()))
        neg_mlh_sem.append(float(neg_mlh.std(ddof=0) / denom))
        if include_vi:
            train_vi = np.array([r["train_mse_vi"] for r in item["retries"]], dtype=np.float64)
            test_vi = np.array([r["test_mse_vi"] for r in item["retries"]], dtype=np.float64)
            neg_vi = -np.array([r["train_log_marglik_vi"] for r in item["retries"]], dtype=np.float64)
            train_mse_vi_mean.append(float(train_vi.mean()))
            train_mse_vi_sem.append(float(train_vi.std(ddof=0) / denom))
            test_mse_vi_mean.append(float(test_vi.mean()))
            test_mse_vi_sem.append(float(test_vi.std(ddof=0) / denom))
            neg_mlh_vi_mean.append(float(neg_vi.mean()))
            neg_mlh_vi_sem.append(float(neg_vi.std(ddof=0) / denom))

    out = {
        "delta": delta_values,
        "train_mse_mean": train_mse_mean,
        "train_mse_sem": train_mse_sem,
        "test_mse_mean": test_mse_mean,
        "test_mse_sem": test_mse_sem,
        "neg_train_marglik_mean": neg_mlh_mean,
        "neg_train_marglik_sem": neg_mlh_sem,
    }
    if include_vi:
        out.update(
            {
                "train_mse_vi_mean": train_mse_vi_mean,
                "train_mse_vi_sem": train_mse_vi_sem,
                "test_mse_vi_mean": test_mse_vi_mean,
                "test_mse_vi_sem": test_mse_vi_sem,
                "neg_train_marglik_vi_mean": neg_mlh_vi_mean,
                "neg_train_marglik_vi_sem": neg_mlh_vi_sem,
            }
        )
    return out


def plot_branch(
    ax_left: plt.Axes,
    delta: np.ndarray,
    train_mse: np.ndarray,
    train_mse_sem: np.ndarray,
    test_mse: np.ndarray,
    test_mse_sem: np.ndarray,
    neg_mlh: np.ndarray,
    neg_mlh_sem: np.ndarray,
    title: str,
    mlh_label: str,
) -> None:
    ax_left.set_xscale("log")
    ax_left.plot(delta, train_mse, "--", color="#7F7F7F", lw=2.0, label="train MSE")
    ax_left.fill_between(delta, train_mse - train_mse_sem, train_mse + train_mse_sem, color="#7F7F7F", alpha=0.2)
    ax_left.plot(delta, test_mse, "-", color="#111111", lw=2.4, label="test MSE")
    ax_left.fill_between(delta, test_mse - test_mse_sem, test_mse + test_mse_sem, color="#111111", alpha=0.12)
    best_test_idx = int(np.argmin(test_mse))
    ax_left.scatter([delta[best_test_idx]], [test_mse[best_test_idx]], marker="*", s=120, color="black", zorder=5)
    ax_left.set_xlabel(r"prior precision $\delta$")
    ax_left.set_ylabel("MSE")

    ax_right = ax_left.twinx()
    ax_right.plot(delta, neg_mlh, color="#1f77b4", lw=2.4, label=mlh_label)
    ax_right.fill_between(delta, neg_mlh - neg_mlh_sem, neg_mlh + neg_mlh_sem, color="#1f77b4", alpha=0.18)
    best_mlh_idx = int(np.argmin(neg_mlh))
    ax_right.scatter([delta[best_mlh_idx]], [neg_mlh[best_mlh_idx]], marker="*", s=120, color="#1f77b4", zorder=5)
    ax_right.set_ylabel("-train log marginal likelihood")

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True, fontsize=9)
    ax_left.set_title(title)


def make_plot(aggregated: Dict[str, List[float]], cfg: Config, fig_path: Path) -> None:
    delta = np.array(aggregated["delta"], dtype=np.float64)
    plt.style.use("seaborn-v0_8-whitegrid")
    if cfg.include_vi and "test_mse_vi_mean" in aggregated:
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharex=True)
        plot_branch(
            axes[0],
            delta,
            np.array(aggregated["train_mse_mean"], dtype=np.float64),
            np.array(aggregated["train_mse_sem"], dtype=np.float64),
            np.array(aggregated["test_mse_mean"], dtype=np.float64),
            np.array(aggregated["test_mse_sem"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_mean"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_sem"], dtype=np.float64),
            "Laplace / DNN2GP",
            "-train log marglik (Laplace)",
        )
        plot_branch(
            axes[1],
            delta,
            np.array(aggregated["train_mse_vi_mean"], dtype=np.float64),
            np.array(aggregated["train_mse_vi_sem"], dtype=np.float64),
            np.array(aggregated["test_mse_vi_mean"], dtype=np.float64),
            np.array(aggregated["test_mse_vi_sem"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_vi_mean"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_vi_sem"], dtype=np.float64),
            "Variational Inference",
            "-train log marglik (VI)",
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2))
        plot_branch(
            ax,
            delta,
            np.array(aggregated["train_mse_mean"], dtype=np.float64),
            np.array(aggregated["train_mse_sem"], dtype=np.float64),
            np.array(aggregated["test_mse_mean"], dtype=np.float64),
            np.array(aggregated["test_mse_sem"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_mean"], dtype=np.float64),
            np.array(aggregated["neg_train_marglik_sem"], dtype=np.float64),
            "Laplace / DNN2GP",
            "-train log marglik (Laplace)",
        )
    fig.suptitle(f"{cfg.dataset_name}: delta tuning", y=0.99, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_outputs(
    cfg: Config,
    device: torch.device,
    raw_results: List[Dict],
    aggregated: Dict[str, List[float]],
    runtime_seconds: float,
) -> Tuple[Path, Path, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.figure_dir / f"marglik_delta_{slugify(cfg.dataset_name)}_{cfg.name}.png"
    json_path = cfg.output_dir / f"delta_sweep_{cfg.name}.json"
    npz_path = cfg.output_dir / f"delta_sweep_{cfg.name}.npz"

    delta = np.array(aggregated["delta"], dtype=np.float64)
    best_test_idx = int(np.argmin(np.array(aggregated["test_mse_mean"], dtype=np.float64)))
    best_mlh_idx = int(np.argmin(np.array(aggregated["neg_train_marglik_mean"], dtype=np.float64)))
    payload = {
        "experiment": "marglik_delta_real",
        "name": cfg.name,
        "dataset": cfg.dataset,
        "dataset_name": cfg.dataset_name,
        "device": str(device),
        "dtype": cfg.dtype,
        "runtime_seconds": runtime_seconds,
        "config": {
            "test_size": cfg.test_size,
            "n_retries": cfg.n_retries,
            "n_params": cfg.n_params,
            "delta_log10_min": cfg.delta_log10_min,
            "delta_log10_max": cfg.delta_log10_max,
            "n_epochs": cfg.n_epochs,
            "vi_epochs": cfg.vi_epochs,
            "include_vi": cfg.include_vi,
            "hidden_size": cfg.hidden_size,
            "n_layers": cfg.n_layers,
            "activation": cfg.activation,
            "sigma_noise": cfg.sigma_noise,
            "learning_rate": cfg.learning_rate,
            "lr_factor": cfg.lr_factor,
            "seed": cfg.seed,
        },
        "best_delta_by_test_mse": float(delta[best_test_idx]),
        "best_delta_by_neg_train_marglik": float(delta[best_mlh_idx]),
        "raw_results": raw_results,
        "aggregated": aggregated,
    }
    if cfg.include_vi and "test_mse_vi_mean" in aggregated:
        best_vi_test = int(np.argmin(np.array(aggregated["test_mse_vi_mean"], dtype=np.float64)))
        best_vi_mlh = int(np.argmin(np.array(aggregated["neg_train_marglik_vi_mean"], dtype=np.float64)))
        payload["best_delta_by_test_mse_vi"] = float(delta[best_vi_test])
        payload["best_delta_by_neg_train_marglik_vi"] = float(delta[best_vi_mlh])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(
        npz_path,
        delta=np.array(aggregated["delta"], dtype=np.float64),
        train_mse_mean=np.array(aggregated["train_mse_mean"], dtype=np.float64),
        train_mse_sem=np.array(aggregated["train_mse_sem"], dtype=np.float64),
        test_mse_mean=np.array(aggregated["test_mse_mean"], dtype=np.float64),
        test_mse_sem=np.array(aggregated["test_mse_sem"], dtype=np.float64),
        neg_train_marglik_mean=np.array(aggregated["neg_train_marglik_mean"], dtype=np.float64),
        neg_train_marglik_sem=np.array(aggregated["neg_train_marglik_sem"], dtype=np.float64),
        train_mse_vi_mean=np.array(aggregated.get("train_mse_vi_mean", []), dtype=np.float64),
        train_mse_vi_sem=np.array(aggregated.get("train_mse_vi_sem", []), dtype=np.float64),
        test_mse_vi_mean=np.array(aggregated.get("test_mse_vi_mean", []), dtype=np.float64),
        test_mse_vi_sem=np.array(aggregated.get("test_mse_vi_sem", []), dtype=np.float64),
        neg_train_marglik_vi_mean=np.array(aggregated.get("neg_train_marglik_vi_mean", []), dtype=np.float64),
        neg_train_marglik_vi_sem=np.array(aggregated.get("neg_train_marglik_vi_sem", []), dtype=np.float64),
    )
    return json_path, npz_path, fig_path


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    splits = make_splits(cfg)
    deltas = np.logspace(cfg.delta_log10_min, cfg.delta_log10_max, cfg.n_params).tolist()

    print("=== real-data delta sweep ===")
    print(
        f"dataset={cfg.dataset} | device={device} | retries={cfg.n_retries} | "
        f"epochs={cfg.n_epochs} | vi_epochs={cfg.vi_epochs} | include_vi={cfg.include_vi}"
    )
    if cfg.include_vi:
        print("note: VI branch uses legacy implementation (CPU-style internals).")
    print(f"deltas: [{deltas[0]:.4g}, ..., {deltas[-1]:.4g}] ({len(deltas)} points)")

    raw_results: List[Dict] = []
    t0 = time.perf_counter()
    for d_ix, delta in enumerate(deltas):
        td = time.perf_counter()
        print(f"\n[{d_ix + 1}/{len(deltas)}] delta={delta:.6g}")
        retries: List[Dict] = []
        for r, split_data in enumerate(splits):
            print(f"  split {r + 1}/{len(splits)} (seed={split_data['split_seed']})")
            metrics = evaluate_delta_on_split(delta, split_data, cfg, device, dtype)
            retries.append(metrics)
            print(
                "    "
                f"test_mse={metrics['test_mse_map']:.4f} | "
                f"train_mse={metrics['train_mse_map']:.4f} | "
                f"log_marglik={metrics['train_log_marglik']:.4f}"
            )
            if cfg.include_vi:
                print(
                    "    "
                    f"[VI] test_mse={metrics['test_mse_vi']:.4f} | "
                    f"train_mse={metrics['train_mse_vi']:.4f} | "
                    f"log_marglik={metrics['train_log_marglik_vi']:.4f}"
                )
        raw_results.append({"delta": float(delta), "retries": retries})
        print(f"  delta elapsed: {(time.perf_counter() - td) / 60.0:.2f} min")

    runtime_seconds = time.perf_counter() - t0
    aggregated = aggregate(raw_results, cfg.include_vi)
    json_path, npz_path, fig_path = save_outputs(cfg, device, raw_results, aggregated, runtime_seconds)
    make_plot(aggregated, cfg, fig_path)

    delta_arr = np.array(aggregated["delta"], dtype=np.float64)
    best_test_idx = int(np.argmin(np.array(aggregated["test_mse_mean"], dtype=np.float64)))
    best_mlh_idx = int(np.argmin(np.array(aggregated["neg_train_marglik_mean"], dtype=np.float64)))

    print("\n=== done ===")
    print(f"runtime: {runtime_seconds / 60.0:.2f} min")
    print(f"best delta by test MSE: {delta_arr[best_test_idx]:.6g}")
    print(f"best delta by -train marglik: {delta_arr[best_mlh_idx]:.6g}")
    if cfg.include_vi and "test_mse_vi_mean" in aggregated:
        best_vi_test = int(np.argmin(np.array(aggregated["test_mse_vi_mean"], dtype=np.float64)))
        best_vi_mlh = int(np.argmin(np.array(aggregated["neg_train_marglik_vi_mean"], dtype=np.float64)))
        print(f"best delta by test MSE (VI): {delta_arr[best_vi_test]:.6g}")
        print(f"best delta by -train marglik (VI): {delta_arr[best_vi_mlh]:.6g}")
    print(f"saved JSON: {json_path}")
    print(f"saved NPZ:  {npz_path}")
    print(f"saved PNG:  {fig_path}")


if __name__ == "__main__":
    run(parse_args())
