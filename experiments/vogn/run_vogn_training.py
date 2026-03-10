from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.datasets import Dataset
from src.vogn import VOGN


class SmallMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: int = 32,
        n_hidden_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        if activation == "tanh":
            act_layer = nn.Tanh
        elif activation == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(act_layer())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_toy_regression_dataset(
    n_train: int = 160,
    n_val: int = 80,
    n_test: int = 320,
    seed: int = 0,
    noise_std: float = 0.12,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)

    x_cand = rng.uniform(-4.0, 4.0, size=n_train * 5)
    x_train = x_cand[(x_cand < -1.0) | (x_cand > 1.0)][:n_train]
    x_train = np.sort(x_train)
    y_train = np.sin(x_train) + rng.normal(0.0, noise_std, size=x_train.shape[0])

    x_val = np.sort(rng.uniform(-4.0, 4.0, size=n_val))
    y_val = np.sin(x_val) + rng.normal(0.0, noise_std, size=n_val)

    x_test = np.linspace(-4.0, 4.0, n_test)
    y_test = np.sin(x_test)

    return {
        "x_train": torch.from_numpy(x_train[:, None]).double(),
        "y_train": torch.from_numpy(y_train).double(),
        "x_val": torch.from_numpy(x_val[:, None]).double(),
        "y_val": torch.from_numpy(y_val).double(),
        "x_test": torch.from_numpy(x_test[:, None]).double(),
        "y_test": torch.from_numpy(y_test).double(),
    }


def toy_regression_loaders(
    data: dict[str, torch.Tensor],
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(data["x_train"], data["y_train"])
    val_ds = TensorDataset(data["x_val"], data["y_val"])
    test_ds = TensorDataset(data["x_test"], data["y_test"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _sample_indices(indices: torch.Tensor, size: int, seed: int) -> list[int]:
    size = min(size, int(indices.numel()))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(indices.numel(), generator=gen)[:size]
    return indices[perm].tolist()


def make_mnist_loaders(
    data_dir: str = "data",
    digits: list[int] | None = None,
    train_size: int = 1200,
    val_size: int = 300,
    test_size: int = 1000,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = Dataset("mnist", data_folder=data_dir)
    train_set = dataset.train_set
    test_set = dataset.test_set

    train_targets = torch.as_tensor(train_set.targets, dtype=torch.long)
    test_targets = torch.as_tensor(test_set.targets, dtype=torch.long)

    if digits is not None:
        digits_t = torch.tensor(digits, dtype=torch.long)
        train_mask = (train_targets[:, None] == digits_t[None, :]).any(dim=1)
        test_mask = (test_targets[:, None] == digits_t[None, :]).any(dim=1)
        train_indices_full = torch.where(train_mask)[0]
        test_indices_full = torch.where(test_mask)[0]
    else:
        train_indices_full = torch.arange(len(train_set))
        test_indices_full = torch.arange(len(test_set))

    train_indices = _sample_indices(train_indices_full, train_size + val_size, seed + 1)
    test_indices = _sample_indices(test_indices_full, test_size, seed + 2)
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(train_set, val_indices)
    test_subset = Subset(test_set, test_indices)

    output_dim = len(digits) if digits is not None else 10
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0),
        output_dim,
    )


def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sqerr_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.double)
            y = y.to(device=device, dtype=torch.double)
            pred = model(x).flatten()
            sqerr_sum += torch.sum((pred - y) ** 2).item()
            n += y.numel()
    return float(sqerr_sum / max(1, n))


def evaluate_classification(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, n = 0.0, 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.double)
            y = y.to(device=device, dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = torch.argmax(logits, dim=1)
            loss_sum += float(loss.item()) * y.numel()
            correct += int((pred == y).sum().item())
            n += y.numel()
    return float(loss_sum / max(1, n)), float(correct / max(1, n))


def _snapshot_state(model: nn.Module, optimizer: VOGN, step: int) -> dict[str, torch.Tensor | int]:
    return {
        "step": int(step),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "mu": optimizer.state["mu"].detach().cpu().clone(),
        "precision": optimizer.state["precision"].detach().cpu().clone(),
    }


def train_regression_variational(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer_class,
    optimizer_name: str,
    device: torch.device,
    total_steps: int = 500,
    checkpoint_steps: list[int] | None = None,
    lr: float = 0.05,
    prior_prec: float = 1.0,
    initial_prec: float = 20.0,
    beta2: float = 0.99,
    num_samples: int = 6,
    eval_interval: int = 10,
) -> dict[str, object]:
    if checkpoint_steps is None:
        checkpoint_steps = [0, 10, 50, 100, 200, 500]

    model = model.to(device=device, dtype=torch.double)
    optimizer = optimizer_class(
        model,
        train_set_size=len(train_loader.dataset),
        lr=lr,
        betas=(0.9, beta2),
        prior_prec=prior_prec,
        inital_prec=initial_prec,
        num_samples=num_samples,
    )

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "test_mse": [],
    }
    checkpoints: dict[int, dict[str, torch.Tensor | int]] = {0: _snapshot_state(model, optimizer, 0)}

    train_iter = cycle(train_loader)
    for step in range(1, total_steps + 1):
        xb, yb = next(train_iter)
        xb = xb.to(device=device, dtype=torch.double)
        yb = yb.to(device=device, dtype=torch.double)

        def closure():
            optimizer.zero_grad()
            pred = model(xb).flatten()
            residuals = torch.abs(pred - yb).detach() + 1e-3
            loss = 0.5 * torch.mean((pred - yb) ** 2)
            return loss, pred, residuals

        loss, _ = optimizer.step(closure)

        if step % eval_interval == 0 or step in checkpoint_steps or step == total_steps:
            train_mse = evaluate_regression(model, train_loader, device)
            val_mse = evaluate_regression(model, val_loader, device)
            test_mse = evaluate_regression(model, test_loader, device)
            history["step"].append(step)
            history["train_loss"].append(train_mse)
            history["val_loss"].append(val_mse)
            history["test_mse"].append(test_mse)

        if step in checkpoint_steps:
            checkpoints[step] = _snapshot_state(model, optimizer, step)

    return {
        "optimizer": optimizer_name,
        "history": history,
        "checkpoints": checkpoints,
        "final_state": _snapshot_state(model, optimizer, total_steps),
    }


def train_classification_variational(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer_class,
    optimizer_name: str,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.03,
    prior_prec: float = 1.0,
    initial_prec: float = 20.0,
    beta2: float = 0.99,
    num_samples: int = 4,
) -> dict[str, object]:
    model = model.to(device=device, dtype=torch.double)
    n_classes = int(model(torch.zeros((1,) + tuple(next(iter(train_loader))[0].shape[1:]), dtype=torch.double, device=device)).shape[1])
    optimizer = optimizer_class(
        model,
        train_set_size=len(train_loader.dataset),
        lr=lr,
        betas=(0.9, beta2),
        prior_prec=prior_prec,
        inital_prec=initial_prec,
        num_samples=num_samples,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_acc": [],
    }
    checkpoints: dict[int, dict[str, torch.Tensor | int]] = {0: _snapshot_state(model, optimizer, 0)}

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.double)
            yb = yb.to(device=device, dtype=torch.long)

            def closure():
                optimizer.zero_grad()
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                one_hot = F.one_hot(yb, num_classes=n_classes).to(dtype=torch.double)
                residuals = torch.sqrt(torch.sum((probs - one_hot) ** 2, dim=1) + 1e-6).detach() + 1e-3
                loss = F.cross_entropy(logits, yb)
                return loss, logits, residuals

            optimizer.step(closure)

        train_loss, _ = evaluate_classification(model, train_loader, device)
        val_loss, _ = evaluate_classification(model, val_loader, device)
        _, test_acc = evaluate_classification(model, test_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_acc"].append(test_acc)
        checkpoints[epoch] = _snapshot_state(model, optimizer, epoch)

    return {
        "optimizer": optimizer_name,
        "history": history,
        "checkpoints": checkpoints,
        "final_state": _snapshot_state(model, optimizer, epochs),
    }


def train_regression_vogn(*args, **kwargs):
    return train_regression_variational(*args, optimizer_class=VOGN, optimizer_name="VOGN", **kwargs)


def train_classification_vogn(*args, **kwargs):
    return train_classification_variational(*args, optimizer_class=VOGN, optimizer_name="VOGN", **kwargs)
