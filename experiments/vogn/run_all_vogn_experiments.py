#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vogn.compute_gp_predictions import (
    gp_predictive_distribution,
    gp_classification_from_qstate,
)
from experiments.vogn.compute_kernels import (
    assign_parameters_from_vector,
    compute_jacobian,
    compute_ntk_kernel,
    compute_oggn_kernel,
    compute_vogn_kernel,
    precision_to_cov_diag,
)
from experiments.vogn.compute_mc_predictions import mc_predict, sample_weights_from_q
from experiments.vogn.plotting_utils import (
    plot_calibration_curve,
    plot_gp_vs_mc,
    plot_kernel_heatmap,
    plot_metric_curves,
    plot_regression_mean_variance,
    plot_uncertainty_histograms,
)
from experiments.vogn.run_oggn_training import (
    train_classification_oggn,
    train_regression_oggn,
)
from experiments.vogn.run_vogn_training import (
    SmallMLP,
    evaluate_classification,
    evaluate_regression,
    make_mnist_loaders,
    make_toy_regression_dataset,
    set_seed,
    toy_regression_loaders,
    train_classification_vogn,
    train_regression_vogn,
)
from experiments.vogn.uncertainty_metrics import (
    binary_auroc,
    brier_score,
    ece_score,
    nll_score,
)


def resolve_device(device_arg: str) -> torch.device:
    req = device_arg.lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto/cuda/cpu")


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def load_snapshot_into_model(
    snapshot: dict,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    hidden_units: int = 32,
    n_hidden_layers: int = 2,
    activation: str = "tanh",
) -> tuple[SmallMLP, dict[str, torch.Tensor]]:
    model = SmallMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_units=hidden_units,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
    ).to(device=device, dtype=torch.double)
    model.load_state_dict(snapshot["model_state"])
    model.eval()
    q_state = {
        "mu": snapshot["mu"].to(device=device, dtype=torch.double),
        "precision": snapshot["precision"].to(device=device, dtype=torch.double),
    }
    return model, q_state


def loader_to_tensors(loader: DataLoader, device: torch.device, dtype: torch.dtype = torch.double):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.to(device=device, dtype=dtype))
        ys.append(y.to(device=device))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def conditioned_gp_regression_from_qstate(
    model: SmallMLP,
    q_state: dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    noise_var: float = 0.02,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        mean_train = model(x_train).flatten()
        mean_test = model(x_test).flatten()

    J_train = compute_jacobian(model, x_train).squeeze(1)
    J_test = compute_jacobian(model, x_test).squeeze(1)
    cov_diag = precision_to_cov_diag(q_state["precision"])

    gp_mean, gp_var, _ = gp_predictive_distribution(
        J_test=J_test,
        J_train=J_train,
        Sigma=cov_diag,
        y_train=y_train,
        noise_var=noise_var,
        mean_train=mean_train,
        mean_test=mean_test,
    )
    K_tt = torch.einsum("np,p,mp->nm", J_train, cov_diag, J_train)
    K_st = torch.einsum("np,p,mp->nm", J_test, cov_diag, J_train)
    eye = torch.eye(K_tt.shape[0], dtype=K_tt.dtype, device=K_tt.device)
    A = K_tt + noise_var * eye
    return {
        "mean": gp_mean,
        "var": gp_var,
        "mean_train": mean_train,
        "mean_test": mean_test,
        "K_st": K_st,
        "A": A,
    }


def conditioned_mc_regression_predict(
    model: SmallMLP,
    q_state: dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    K_st: torch.Tensor,
    A: torch.Tensor,
    mc_samples: int = 80,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    mu = q_state["mu"].to(device=x_test.device, dtype=x_test.dtype)
    precision = q_state["precision"].to(device=x_test.device, dtype=x_test.dtype)
    weight_samples = sample_weights_from_q(mu=mu, precision=precision, n_samples=mc_samples, seed=seed)

    preds = []
    with torch.no_grad():
        for w in weight_samples:
            assign_parameters_from_vector(model, w)
            m_train = model(x_train).flatten()
            m_test = model(x_test).flatten()
            alpha = torch.linalg.solve(A, y_train - m_train)
            preds.append(m_test + K_st @ alpha)

    assign_parameters_from_vector(model, mu)
    pred_samples = torch.stack(preds, dim=0)
    return {
        "mean": pred_samples.mean(dim=0).cpu().numpy(),
        "var": pred_samples.var(dim=0).clamp_min(1e-12).cpu().numpy(),
        "samples": pred_samples.cpu().numpy(),
    }


def train_regression_deterministic(
    optimizer_name: str,
    model: SmallMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    total_steps: int = 250,
    lr: float = 0.01,
    eval_interval: int = 10,
) -> dict:
    model = model.to(device=device, dtype=torch.double)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_name must be Adam or RMSprop")

    train_iter = cycle(train_loader)
    history = {"step": [], "train_loss": [], "val_loss": [], "test_mse": []}
    checkpoints = {0: {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}}

    for step in range(1, total_steps + 1):
        xb, yb = next(train_iter)
        xb = xb.to(device=device, dtype=torch.double)
        yb = yb.to(device=device, dtype=torch.double)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb).flatten()
        loss = 0.5 * torch.mean((pred - yb) ** 2)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == total_steps:
            history["step"].append(step)
            history["train_loss"].append(evaluate_regression(model, train_loader, device))
            history["val_loss"].append(evaluate_regression(model, val_loader, device))
            history["test_mse"].append(evaluate_regression(model, test_loader, device))

    checkpoints[total_steps] = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return {"optimizer": optimizer_name, "history": history, "checkpoints": checkpoints}


def train_classification_deterministic(
    optimizer_name: str,
    model: SmallMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 6,
    lr: float = 1e-3,
) -> dict:
    model = model.to(device=device, dtype=torch.double)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_name must be Adam or RMSprop")

    history = {"epoch": [], "train_loss": [], "val_loss": [], "test_acc": []}
    checkpoints = {0: {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}}

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.double)
            yb = yb.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        tr_loss, _ = evaluate_classification(model, train_loader, device)
        va_loss, _ = evaluate_classification(model, val_loader, device)
        _, te_acc = evaluate_classification(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["test_acc"].append(te_acc)
        checkpoints[epoch] = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    return {"optimizer": optimizer_name, "history": history, "checkpoints": checkpoints}


def experiment1_gp_evolution(root: Path, device: torch.device, seed: int) -> tuple[dict, dict]:
    exp_dir = root / "experiment1_gp_evolution"
    (exp_dir / "vogn").mkdir(parents=True, exist_ok=True)
    (exp_dir / "oggn").mkdir(parents=True, exist_ok=True)

    data = make_toy_regression_dataset(
        n_train=200,
        n_val=80,
        n_test=320,
        seed=seed,
        noise_std=0.08,
    )
    train_loader, val_loader, test_loader = toy_regression_loaders(data, batch_size=32)
    checkpoint_steps = [0, 100, 250, 500, 1000, 1750, 2500]
    gp_noise_var = 0.02

    vogn_model = SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh")
    oggn_model = SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh")

    vogn_res = train_regression_vogn(
        model=vogn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        total_steps=2500,
        checkpoint_steps=checkpoint_steps,
        lr=0.03,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=6,
        eval_interval=50,
    )
    oggn_res = train_regression_oggn(
        model=oggn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        total_steps=2500,
        checkpoint_steps=checkpoint_steps,
        lr=0.03,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=6,
        eval_interval=50,
    )

    x_train_t = data["x_train"].to(device)
    y_train_t = data["y_train"].to(device)
    y_test_t = data["y_test"].to(device)
    x_test = data["x_test"].to(device)
    x_train_np = data["x_train"].cpu().numpy().squeeze()
    y_train_np = data["y_train"].cpu().numpy().squeeze()
    x_test_np = data["x_test"].cpu().numpy().squeeze()

    metrics = {"VOGN": {}, "OGGN": {}}
    for name, res in [("VOGN", vogn_res), ("OGGN", oggn_res)]:
        opt_dir = exp_dir / name.lower()
        for step in checkpoint_steps:
            snap = res["checkpoints"][step]
            model_step, q_state = load_snapshot_into_model(snap, input_dim=1, output_dim=1, device=device)
            gp_out = conditioned_gp_regression_from_qstate(
                model=model_step,
                q_state=q_state,
                x_train=x_train_t,
                y_train=y_train_t,
                x_test=x_test,
                noise_var=gp_noise_var,
            )
            mean = gp_out["mean"].detach().cpu().numpy()
            var = gp_out["var"].detach().cpu().numpy()
            test_mse = float(torch.mean((gp_out["mean"] - y_test_t) ** 2).item())

            np.savez(
                opt_dir / f"gp_step_{step:03d}.npz",
                x_test=x_test_np,
                mean=mean,
                var=var,
                x_train=x_train_np,
                y_train=y_train_np,
            )
            plot_regression_mean_variance(
                x_train=x_train_np,
                y_train=y_train_np,
                x_test=x_test_np,
                mean=mean,
                var=var,
                title=f"{name} posterior GP evolution (step {step})",
                path=opt_dir / f"mean_variance_step_{step:03d}.png",
            )
            metrics[name][f"step_{step}"] = {
                "mean_abs_mean": float(np.mean(np.abs(mean))),
                "mean_var": float(np.mean(var)),
                "max_var": float(np.max(var)),
                "test_mse_posterior_mean": test_mse,
            }

    save_json(exp_dir / "metrics.json", metrics)
    return metrics, {
        "data": data,
        "checkpoint_steps": checkpoint_steps,
        "gp_noise_var": gp_noise_var,
        "vogn": vogn_res,
        "oggn": oggn_res,
    }


def experiment2_optimizer_comparison(root: Path, device: torch.device, seed: int) -> tuple[dict, dict]:
    exp_dir = root / "experiment2_optimizer_comparison"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Regression comparison
    toy = make_toy_regression_dataset(seed=seed + 11)
    tr_reg, va_reg, te_reg = toy_regression_loaders(toy, batch_size=32)

    reg_hist = {}
    reg_hist["VOGN"] = train_regression_vogn(
        model=SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh"),
        train_loader=tr_reg,
        val_loader=va_reg,
        test_loader=te_reg,
        device=device,
        total_steps=1500,
        checkpoint_steps=[0, 250, 500, 1000, 1500],
        lr=0.03,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=4,
        eval_interval=25,
    )
    reg_hist["OGGN"] = train_regression_oggn(
        model=SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh"),
        train_loader=tr_reg,
        val_loader=va_reg,
        test_loader=te_reg,
        device=device,
        total_steps=1500,
        checkpoint_steps=[0, 250, 500, 1000, 1500],
        lr=0.03,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=4,
        eval_interval=25,
    )
    reg_hist["Adam"] = train_regression_deterministic(
        optimizer_name="Adam",
        model=SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh"),
        train_loader=tr_reg,
        val_loader=va_reg,
        test_loader=te_reg,
        device=device,
        total_steps=1500,
        lr=0.005,
        eval_interval=25,
    )
    reg_hist["RMSprop"] = train_regression_deterministic(
        optimizer_name="RMSprop",
        model=SmallMLP(1, 1, hidden_units=32, n_hidden_layers=2, activation="tanh"),
        train_loader=tr_reg,
        val_loader=va_reg,
        test_loader=te_reg,
        device=device,
        total_steps=1500,
        lr=0.005,
        eval_interval=25,
    )

    x_steps = np.array(reg_hist["VOGN"]["history"]["step"])
    plot_metric_curves(
        x_values=x_steps,
        series={k: np.array(v["history"]["train_loss"]) for k, v in reg_hist.items()},
        title="Regression training loss vs iteration",
        y_label="MSE",
        path=exp_dir / "regression_loss_vs_iteration.png",
        x_label="Iteration",
    )
    plot_metric_curves(
        x_values=x_steps,
        series={k: np.array(v["history"]["test_mse"]) for k, v in reg_hist.items()},
        title="Regression test MSE vs iteration",
        y_label="Test MSE",
        path=exp_dir / "regression_test_mse_vs_iteration.png",
        x_label="Iteration",
    )

    # Binary MNIST comparison
    tr_bin, va_bin, te_bin, out_dim_bin = make_mnist_loaders(
        data_dir="data",
        digits=[0, 1],
        train_size=2500,
        val_size=500,
        test_size=1200,
        batch_size=64,
        seed=seed + 22,
    )

    clf_hist = {}
    clf_hist["VOGN"] = train_classification_vogn(
        model=SmallMLP(28 * 28, out_dim_bin, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=tr_bin,
        val_loader=va_bin,
        test_loader=te_bin,
        device=device,
        epochs=12,
        lr=0.01,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=3,
    )
    clf_hist["OGGN"] = train_classification_oggn(
        model=SmallMLP(28 * 28, out_dim_bin, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=tr_bin,
        val_loader=va_bin,
        test_loader=te_bin,
        device=device,
        epochs=12,
        lr=0.01,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=3,
    )
    clf_hist["Adam"] = train_classification_deterministic(
        optimizer_name="Adam",
        model=SmallMLP(28 * 28, out_dim_bin, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=tr_bin,
        val_loader=va_bin,
        test_loader=te_bin,
        device=device,
        epochs=12,
        lr=3e-4,
    )
    clf_hist["RMSprop"] = train_classification_deterministic(
        optimizer_name="RMSprop",
        model=SmallMLP(28 * 28, out_dim_bin, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=tr_bin,
        val_loader=va_bin,
        test_loader=te_bin,
        device=device,
        epochs=12,
        lr=3e-4,
    )

    x_epochs = np.array(clf_hist["VOGN"]["history"]["epoch"])
    steps_per_epoch = len(tr_bin)
    x_iterations = x_epochs * steps_per_epoch
    plot_metric_curves(
        x_values=x_iterations,
        series={k: np.array(v["history"]["train_loss"]) for k, v in clf_hist.items()},
        title="Binary MNIST training loss vs iteration",
        y_label="Cross-entropy",
        path=exp_dir / "classification_loss_vs_iteration.png",
        x_label="Iteration",
    )
    plot_metric_curves(
        x_values=x_iterations,
        series={k: np.array(v["history"]["test_acc"]) for k, v in clf_hist.items()},
        title="Binary MNIST test accuracy vs iteration",
        y_label="Test accuracy",
        path=exp_dir / "classification_accuracy_vs_iteration.png",
        x_label="Iteration",
    )

    reg_arrays = {"steps": x_steps}
    for opt_name, run in reg_hist.items():
        reg_arrays[f"{opt_name.lower()}_train_loss"] = np.array(run["history"]["train_loss"])
        reg_arrays[f"{opt_name.lower()}_val_loss"] = np.array(run["history"]["val_loss"])
        reg_arrays[f"{opt_name.lower()}_test_mse"] = np.array(run["history"]["test_mse"])
    np.savez(exp_dir / "regression_histories.npz", **reg_arrays)

    clf_arrays = {"epochs": x_epochs, "iterations": x_iterations}
    for opt_name, run in clf_hist.items():
        clf_arrays[f"{opt_name.lower()}_train_loss"] = np.array(run["history"]["train_loss"])
        clf_arrays[f"{opt_name.lower()}_val_loss"] = np.array(run["history"]["val_loss"])
        clf_arrays[f"{opt_name.lower()}_test_acc"] = np.array(run["history"]["test_acc"])
    np.savez(exp_dir / "classification_histories.npz", **clf_arrays)

    metrics = {
        "regression": {
            k: {
                "final_train_loss": float(v["history"]["train_loss"][-1]),
                "final_val_loss": float(v["history"]["val_loss"][-1]),
                "final_test_mse": float(v["history"]["test_mse"][-1]),
            }
            for k, v in reg_hist.items()
        },
        "classification": {
            k: {
                "final_train_loss": float(v["history"]["train_loss"][-1]),
                "final_val_loss": float(v["history"]["val_loss"][-1]),
                "final_test_acc": float(v["history"]["test_acc"][-1]),
            }
            for k, v in clf_hist.items()
        },
    }
    save_json(exp_dir / "metrics.json", metrics)
    return metrics, {"reg_hist": reg_hist, "clf_hist": clf_hist, "binary_dim": out_dim_bin}


def experiment3_gp_vs_mc(root: Path, device: torch.device, exp1_artifacts: dict) -> dict:
    exp_dir = root / "experiment3_gp_vs_mc"
    exp_dir.mkdir(parents=True, exist_ok=True)

    data = exp1_artifacts["data"]
    x_train = data["x_train"].to(device)
    y_train = data["y_train"].to(device)
    x_test = data["x_test"].to(device)
    y_test = data["y_test"].to(device)
    x_test_np = data["x_test"].cpu().numpy().squeeze()
    x_train_np = data["x_train"].cpu().numpy().squeeze()
    y_train_np = data["y_train"].cpu().numpy().squeeze()
    y_test_np = data["y_test"].cpu().numpy().squeeze()
    checkpoint_steps = exp1_artifacts["checkpoint_steps"]
    gp_noise_var = float(exp1_artifacts.get("gp_noise_var", 0.02))

    metrics = {"VOGN": {}, "OGGN": {}}
    for name, res in [("VOGN", exp1_artifacts["vogn"]), ("OGGN", exp1_artifacts["oggn"])]:
        for step in checkpoint_steps:
            snap = res["checkpoints"][step]
            model_step, q_state = load_snapshot_into_model(snap, input_dim=1, output_dim=1, device=device)
            gp_terms = conditioned_gp_regression_from_qstate(
                model=model_step,
                q_state=q_state,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                noise_var=gp_noise_var,
            )
            mc_out = conditioned_mc_regression_predict(
                model=model_step,
                q_state=q_state,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                K_st=gp_terms["K_st"],
                A=gp_terms["A"],
                mc_samples=60,
                seed=step + 17,
            )

            gp_mean = gp_terms["mean"].detach().cpu().numpy()
            gp_var = gp_terms["var"].detach().cpu().numpy()
            mc_mean, mc_var = mc_out["mean"], mc_out["var"]
            mean_mse = float(np.mean((gp_mean - mc_mean) ** 2))
            var_mse = float(np.mean((gp_var - mc_var) ** 2))
            gp_test_mse = float(torch.mean((gp_terms["mean"] - y_test) ** 2).item())
            mc_test_mse = float(np.mean((mc_mean - y_test_np) ** 2))
            metrics[name][f"step_{step}"] = {
                "mean_mse": mean_mse,
                "var_mse": var_mse,
                "gp_test_mse": gp_test_mse,
                "mc_test_mse": mc_test_mse,
            }

            out_prefix = exp_dir / f"{name.lower()}_step_{step:03d}"
            np.savez(
                out_prefix.with_suffix(".npz"),
                x_test=x_test_np,
                y_test=y_test_np,
                gp_mean=gp_mean,
                gp_var=gp_var,
                mc_mean=mc_mean,
                mc_var=mc_var,
            )
            plot_gp_vs_mc(
                x_train=x_train_np,
                y_train=y_train_np,
                x_test=x_test_np,
                gp_mean=gp_mean,
                gp_var=gp_var,
                mc_mean=mc_mean,
                mc_var=mc_var,
                title=f"{name}: conditioned GP vs conditioned MC at step {step}",
                path=exp_dir / f"{name.lower()}_gp_vs_mc_step_{step:03d}.png",
            )

    save_json(exp_dir / "metrics.json", metrics)
    return metrics


def experiment4_kernel_evolution(root: Path, device: torch.device, exp1_artifacts: dict) -> dict:
    exp_dir = root / "experiment4_kernel_evolution"
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_steps = exp1_artifacts["checkpoint_steps"]
    x_kernel = torch.linspace(-4.0, 4.0, 80, device=device, dtype=torch.double)[:, None]
    kernel_cache: dict[int, dict[str, np.ndarray]] = {}

    for step in checkpoint_steps:
        snap_v = exp1_artifacts["vogn"]["checkpoints"][step]
        snap_o = exp1_artifacts["oggn"]["checkpoints"][step]

        model_v, q_v = load_snapshot_into_model(snap_v, input_dim=1, output_dim=1, device=device)
        model_o, q_o = load_snapshot_into_model(snap_o, input_dim=1, output_dim=1, device=device)

        J_v = compute_jacobian(model_v, x_kernel).squeeze(1)
        J_o = compute_jacobian(model_o, x_kernel).squeeze(1)

        cov_v = precision_to_cov_diag(q_v["precision"])
        cov_o = precision_to_cov_diag(q_o["precision"])

        K_ntk = compute_ntk_kernel(J_v).detach().cpu().numpy()
        K_vogn = compute_vogn_kernel(J_v, cov_v).detach().cpu().numpy()
        K_oggn = compute_oggn_kernel(J_o, cov_o).detach().cpu().numpy()
        kernel_cache[step] = {"ntk": K_ntk, "vogn": K_vogn, "oggn": K_oggn}

    ntk_vmin = float(min(np.min(kernel_cache[step]["ntk"]) for step in checkpoint_steps))
    ntk_vmax = float(max(np.max(kernel_cache[step]["ntk"]) for step in checkpoint_steps))
    vogn_vmin = float(min(np.min(kernel_cache[step]["vogn"]) for step in checkpoint_steps))
    vogn_vmax = float(max(np.max(kernel_cache[step]["vogn"]) for step in checkpoint_steps))
    oggn_vmin = float(min(np.min(kernel_cache[step]["oggn"]) for step in checkpoint_steps))
    oggn_vmax = float(max(np.max(kernel_cache[step]["oggn"]) for step in checkpoint_steps))
    base_step = checkpoint_steps[0]
    K_ntk_0 = kernel_cache[base_step]["ntk"]
    K_vogn_0 = kernel_cache[base_step]["vogn"]
    K_oggn_0 = kernel_cache[base_step]["oggn"]

    metrics = {}
    for step in checkpoint_steps:
        K_ntk = kernel_cache[step]["ntk"]
        K_vogn = kernel_cache[step]["vogn"]
        K_oggn = kernel_cache[step]["oggn"]

        np.save(exp_dir / f"kernel_ntk_step_{step:03d}.npy", K_ntk)
        np.save(exp_dir / f"kernel_vogn_step_{step:03d}.npy", K_vogn)
        np.save(exp_dir / f"kernel_oggn_step_{step:03d}.npy", K_oggn)

        plot_kernel_heatmap(
            K_ntk,
            f"NTK-like kernel (step {step})",
            exp_dir / f"kernel_ntk_step_{step:03d}.png",
            vmin=ntk_vmin,
            vmax=ntk_vmax,
        )
        plot_kernel_heatmap(
            K_vogn,
            f"VOGN kernel (step {step})",
            exp_dir / f"kernel_vogn_step_{step:03d}.png",
            vmin=vogn_vmin,
            vmax=vogn_vmax,
        )
        plot_kernel_heatmap(
            K_oggn,
            f"OGGN kernel (step {step})",
            exp_dir / f"kernel_oggn_step_{step:03d}.png",
            vmin=oggn_vmin,
            vmax=oggn_vmax,
        )

        metrics[f"step_{step}"] = {
            "trace_ntk": float(np.trace(K_ntk)),
            "trace_vogn": float(np.trace(K_vogn)),
            "trace_oggn": float(np.trace(K_oggn)),
            "fro_delta_ntk": float(np.linalg.norm(K_ntk - K_ntk_0)),
            "fro_delta_vogn": float(np.linalg.norm(K_vogn - K_vogn_0)),
            "fro_delta_oggn": float(np.linalg.norm(K_oggn - K_oggn_0)),
        }

    save_json(exp_dir / "metrics.json", metrics)
    return metrics


def experiment5_ood_detection(root: Path, device: torch.device, exp2_artifacts: dict, seed: int) -> dict:
    exp_dir = root / "experiment5_ood_detection"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Use full MNIST test labels (0-9), sampled subset for tractability.
    _, _, test_loader_all, _ = make_mnist_loaders(
        data_dir="data",
        digits=None,
        train_size=10,
        val_size=10,
        test_size=3000,
        batch_size=256,
        seed=seed + 333,
    )
    x_all, y_all = loader_to_tensors(test_loader_all, device=device, dtype=torch.double)
    y_all_np = y_all.detach().cpu().numpy().astype(np.int64)
    seen_mask = np.isin(y_all_np, [0, 1])
    ood_labels = (~seen_mask).astype(np.int64)

    metrics = {}
    for name in ["VOGN", "OGGN"]:
        final_state = exp2_artifacts["clf_hist"][name]["final_state"]
        model, q_state = load_snapshot_into_model(final_state, input_dim=28 * 28, output_dim=2, device=device, activation="relu")
        out = gp_classification_from_qstate(model, x_all, q_state, mc_samples=80, seed=13)

        entropy = out["entropy"]
        pred_var = out["predictive_var"]
        auroc_entropy = binary_auroc(entropy, ood_labels)
        auroc_var = binary_auroc(pred_var, ood_labels)

        seen_entropy = entropy[seen_mask]
        unseen_entropy = entropy[~seen_mask]
        seen_var = pred_var[seen_mask]
        unseen_var = pred_var[~seen_mask]

        np.savez(
            exp_dir / f"{name.lower()}_ood_arrays.npz",
            entropy=entropy,
            pred_var=pred_var,
            labels=y_all_np,
            ood_labels=ood_labels,
        )
        plot_uncertainty_histograms(
            seen_scores=seen_entropy,
            unseen_scores=unseen_entropy,
            title=f"{name} OOD detection: predictive entropy",
            x_label="Predictive entropy",
            path=exp_dir / f"{name.lower()}_ood_entropy.png",
        )
        plot_uncertainty_histograms(
            seen_scores=seen_var,
            unseen_scores=unseen_var,
            title=f"{name} OOD detection: predictive variance",
            x_label="Predictive variance",
            path=exp_dir / f"{name.lower()}_ood_variance.png",
        )

        metrics[name] = {
            "auroc_entropy": float(auroc_entropy),
            "auroc_predictive_variance": float(auroc_var),
            "mean_entropy_seen": float(np.mean(seen_entropy)),
            "mean_entropy_unseen": float(np.mean(unseen_entropy)),
            "mean_var_seen": float(np.mean(seen_var)),
            "mean_var_unseen": float(np.mean(unseen_var)),
        }

    save_json(exp_dir / "metrics.json", metrics)
    return metrics


def experiment6_calibration(root: Path, device: torch.device, seed: int) -> dict:
    exp_dir = root / "experiment6_calibration"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, out_dim = make_mnist_loaders(
        data_dir="data",
        digits=None,
        train_size=3000,
        val_size=500,
        test_size=1000,
        batch_size=128,
        seed=seed + 444,
    )
    x_test, y_test = loader_to_tensors(test_loader, device=device, dtype=torch.double)
    y_test_np = y_test.detach().cpu().numpy().astype(np.int64)

    vogn_res = train_classification_vogn(
        model=SmallMLP(28 * 28, out_dim, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=12,
        lr=0.01,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=3,
    )
    oggn_res = train_classification_oggn(
        model=SmallMLP(28 * 28, out_dim, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=12,
        lr=0.01,
        prior_prec=1.0,
        initial_prec=20.0,
        beta2=0.99,
        num_samples=3,
    )
    adam_res = train_classification_deterministic(
        optimizer_name="Adam",
        model=SmallMLP(28 * 28, out_dim, hidden_units=32, n_hidden_layers=2, activation="relu"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=12,
        lr=3e-4,
    )

    all_runs = {"VOGN": vogn_res, "OGGN": oggn_res, "Adam": adam_res}
    calibration_metrics = {"NLL": {}, "ECE": {}, "Brier": {}}
    final_probs = {}

    for name, run in all_runs.items():
        nll_curve, ece_curve, brier_curve = [], [], []
        epoch_keys = sorted([k for k in run["checkpoints"].keys() if isinstance(k, int) and k > 0])
        for ep in epoch_keys:
            snap = run["checkpoints"][ep]
            if name in {"VOGN", "OGGN"}:
                model_ep, q_state = load_snapshot_into_model(
                    snap,
                    input_dim=28 * 28,
                    output_dim=out_dim,
                    device=device,
                    activation="relu",
                )
                out = mc_predict(model_ep, q_state, x_test, mc_samples=40, classification=True, seed=ep + 123)
                probs = out["probs"]
            else:
                model_ep = SmallMLP(28 * 28, out_dim, hidden_units=32, n_hidden_layers=2, activation="relu").to(
                    device=device, dtype=torch.double
                )
                model_ep.load_state_dict(snap["model_state"])
                model_ep.eval()
                with torch.no_grad():
                    probs = torch.softmax(model_ep(x_test), dim=1).cpu().numpy()

            nll_curve.append(nll_score(probs, y_test_np))
            ece_curve.append(ece_score(probs, y_test_np, n_bins=15))
            brier_curve.append(brier_score(probs, y_test_np))
            if ep == epoch_keys[-1]:
                final_probs[name] = probs

        calibration_metrics["NLL"][name] = nll_curve
        calibration_metrics["ECE"][name] = ece_curve
        calibration_metrics["Brier"][name] = brier_curve

    x_epochs = np.arange(1, len(calibration_metrics["NLL"]["VOGN"]) + 1)
    plot_metric_curves(
        x_values=x_epochs,
        series={k: np.array(v) for k, v in calibration_metrics["NLL"].items()},
        title="NLL during training",
        y_label="NLL",
        path=exp_dir / "nll_vs_epoch.png",
        x_label="Epoch",
    )
    plot_metric_curves(
        x_values=x_epochs,
        series={k: np.array(v) for k, v in calibration_metrics["ECE"].items()},
        title="ECE during training",
        y_label="ECE",
        path=exp_dir / "ece_vs_epoch.png",
        x_label="Epoch",
    )
    plot_metric_curves(
        x_values=x_epochs,
        series={k: np.array(v) for k, v in calibration_metrics["Brier"].items()},
        title="Brier score during training",
        y_label="Brier",
        path=exp_dir / "brier_vs_epoch.png",
        x_label="Epoch",
    )

    for name, probs in final_probs.items():
        plot_calibration_curve(
            probs=probs,
            labels=y_test_np,
            title=f"Calibration curve ({name}, final epoch)",
            path=exp_dir / f"calibration_curve_{name.lower()}.png",
            n_bins=15,
        )

    calibration_arrays = {"y_test": y_test_np}
    for metric_name, per_model in calibration_metrics.items():
        for model_name, values in per_model.items():
            calibration_arrays[f"{metric_name.lower()}_{model_name.lower()}"] = np.array(values)
    for model_name, probs in final_probs.items():
        calibration_arrays[f"final_probs_{model_name.lower()}"] = probs
    np.savez(exp_dir / "calibration_arrays.npz", **calibration_arrays)

    save_json(exp_dir / "metrics.json", calibration_metrics)
    return calibration_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all VOGN/OGGN experiments.")
    parser.add_argument("--results-root", type=str, default="results/vogn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Saving all outputs under: {results_root}")

    summary = {}

    print("\n[Experiment 1] GP evolution")
    m1, exp1_artifacts = experiment1_gp_evolution(results_root, device=device, seed=args.seed)
    summary["experiment1_gp_evolution"] = m1

    print("\n[Experiment 2] Optimizer comparison")
    m2, exp2_artifacts = experiment2_optimizer_comparison(results_root, device=device, seed=args.seed)
    summary["experiment2_optimizer_comparison"] = m2

    print("\n[Experiment 3] GP vs MC")
    m3 = experiment3_gp_vs_mc(results_root, device=device, exp1_artifacts=exp1_artifacts)
    summary["experiment3_gp_vs_mc"] = m3

    print("\n[Experiment 4] Kernel evolution")
    m4 = experiment4_kernel_evolution(results_root, device=device, exp1_artifacts=exp1_artifacts)
    summary["experiment4_kernel_evolution"] = m4

    print("\n[Experiment 5] OOD detection")
    m5 = experiment5_ood_detection(results_root, device=device, exp2_artifacts=exp2_artifacts, seed=args.seed)
    summary["experiment5_ood_detection"] = m5

    print("\n[Experiment 6] Calibration")
    m6 = experiment6_calibration(results_root, device=device, seed=args.seed)
    summary["experiment6_calibration"] = m6

    save_json(results_root / "summary_metrics.json", summary)
    print("\nAll experiments complete.")
    print(f"Summary file: {results_root / 'summary_metrics.json'}")

if __name__ == "__main__":
    main()
