from __future__ import annotations

import argparse
import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import Dataset
from src.dnn2gp import compute_laplace
from src.neural_networks import LeNet5


@dataclass
class ModelOutputs:
    y_true: np.ndarray
    y_pred: np.ndarray
    uncertainty: np.ndarray
    confidence: np.ndarray

    @property
    def correct(self) -> np.ndarray:
        return (self.y_true == self.y_pred).astype(np.int64)


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
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError("Unsupported device; use one of: auto, cuda, cpu.")


def parse_coverage_levels(raw: str) -> np.ndarray:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    arr = np.array(values, dtype=np.float64)
    if np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError("Coverage values must be in (0, 1].")
    return np.sort(arr)


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def assign_parameters_from_vector(model: nn.Module, vector: torch.Tensor) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[idx : idx + n].view_as(p))
            idx += n


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    n_items = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        loss_sum += float(loss.item()) * bs
        n_items += bs
    return loss_sum / max(1, n_items)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_total = 0
    n_correct = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        pred = model(x).argmax(dim=1)
        n_correct += int((pred == y).sum().item())
        n_total += int(y.numel())
    return n_correct / max(1, n_total)


def load_or_train_map_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    force_train: bool,
) -> nn.Module:
    model = LeNet5(input_channels=1, dims=28, num_classes=10).to(device=device, dtype=torch.float32)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not force_train:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Loaded MAP checkpoint: {checkpoint_path} | test_acc={100*acc:.2f}%")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | test_acc={100*test_acc:.2f}%")

    final_acc = evaluate_accuracy(model, test_loader, device)
    torch.save({"model_state": model.state_dict(), "test_accuracy": final_acc}, checkpoint_path)
    print(f"Saved MAP checkpoint: {checkpoint_path} | test_acc={100*final_acc:.2f}%")
    return model


def compute_or_load_laplace_precision(
    model_double: nn.Module,
    train_set,
    device: torch.device,
    cache_path: Path,
    prior_prec: float,
    laplace_train_size: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached Laplace precision: {cache_path}")
        return torch.load(cache_path, map_location=device).to(device=device, dtype=torch.double)

    n_total = len(train_set)
    subset_size = min(laplace_train_size, n_total)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=rng)[:subset_size].tolist()
    subset = Subset(train_set, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        "Computing Laplace diagonal precision on MNIST train subset "
        f"(n={subset_size}, prior_prec={prior_prec}, batch_size={batch_size})"
    )
    post_prec = compute_laplace(model=model_double, train_loader=loader, prior_prec=prior_prec, device=device)
    post_prec = post_prec.detach().to(device=device, dtype=torch.double)
    torch.save(post_prec.detach().cpu(), cache_path)
    print(f"Saved Laplace precision cache: {cache_path}")
    return post_prec


@torch.no_grad()
def predict_standard_nn(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    uncertainty_type: str = "entropy",
) -> ModelOutputs:
    model.eval()
    y_true, y_pred, unc, conf = [], [], [], []
    for x, y in test_loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        probs = torch.softmax(model(x), dim=1)
        pred = torch.argmax(probs, dim=1)
        max_prob = probs.max(dim=1).values
        if uncertainty_type == "entropy":
            uncertainty = entropy_from_probs(probs)
        elif uncertainty_type == "one_minus_max":
            uncertainty = 1.0 - max_prob
        else:
            raise ValueError("Unsupported uncertainty type.")

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        unc.append(uncertainty.cpu().numpy())
        conf.append(max_prob.cpu().numpy())

    return ModelOutputs(
        y_true=np.concatenate(y_true),
        y_pred=np.concatenate(y_pred),
        uncertainty=np.concatenate(unc),
        confidence=np.concatenate(conf),
    )


def predict_laplace_mc(
    model_double: nn.Module,
    test_loader: DataLoader,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
) -> ModelOutputs:
    model_double.eval()
    theta_map = flatten_parameters(model_double)
    std = torch.rsqrt(post_prec.clamp_min(1e-12))
    params = [p for p in model_double.parameters()]
    generator = torch.Generator(device=device).manual_seed(12345)

    y_true, y_pred, unc, conf = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Laplace MC inference"):
            x = x.to(device=device, dtype=torch.double)
            y = y.to(device=device, dtype=torch.long)
            sum_probs = torch.zeros((x.shape[0], 10), device=device, dtype=torch.double)
            for _ in range(mc_samples):
                eps = torch.randn(theta_map.shape, generator=generator, device=device, dtype=torch.double)
                theta_s = theta_map + std * eps
                assign_parameters_from_vector(model_double, theta_s)
                sum_probs += torch.softmax(model_double(x), dim=1)
            assign_parameters_from_vector(model_double, theta_map)

            probs = sum_probs / mc_samples
            pred = torch.argmax(probs, dim=1)
            max_prob = probs.max(dim=1).values
            uncertainty = entropy_from_probs(probs)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            unc.append(uncertainty.cpu().numpy())
            conf.append(max_prob.cpu().numpy())

    return ModelOutputs(
        y_true=np.concatenate(y_true),
        y_pred=np.concatenate(y_pred),
        uncertainty=np.concatenate(unc),
        confidence=np.concatenate(conf),
    )


def _jacobian_logits_wrt_params(
    model_double: nn.Module,
    x_single: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model_double(x_single).squeeze(0)
    params = [p for p in model_double.parameters()]
    jac_rows = []
    for k in range(logits.shape[0]):
        retain = k < logits.shape[0] - 1
        grads = torch.autograd.grad(logits[k], params, retain_graph=retain, create_graph=False, allow_unused=False)
        jac_rows.append(torch.cat([g.reshape(-1) for g in grads], dim=0))
    return logits.detach(), torch.stack(jac_rows, dim=0).detach()


def predict_dnn2gp_gp_sampling(
    model_double: nn.Module,
    test_loader: DataLoader,
    post_prec: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    diag_cov: bool = True,
) -> ModelOutputs:
    model_double.eval()
    inv_prec = torch.reciprocal(post_prec.clamp_min(1e-12))
    y_true, y_pred, unc, conf = [], [], [], []
    eye = torch.eye(10, dtype=torch.double, device=device)
    generator = torch.Generator(device=device).manual_seed(54321)

    for x, y in tqdm(test_loader, desc="DNN2GP GP inference"):
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.long)
        if x.shape[0] != 1:
            raise ValueError("DNN2GP inference loader must use batch_size=1.")

        mu, jac = _jacobian_logits_wrt_params(model_double, x)
        if diag_cov:
            var_diag = torch.einsum("kp,p,kp->k", jac, inv_prec, jac).clamp_min(1e-12)
            eps = torch.randn((mc_samples, 10), generator=generator, device=device, dtype=torch.double)
            logits_samples = mu.unsqueeze(0) + eps * torch.sqrt(var_diag).unsqueeze(0)
        else:
            cov = torch.einsum("kp,p,mp->km", jac, inv_prec, jac)
            cov = 0.5 * (cov + cov.T) + 1e-6 * eye
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
            logits_samples = dist.rsample((mc_samples,))

        probs = torch.softmax(logits_samples, dim=1).mean(dim=0)
        pred = int(torch.argmax(probs).item())
        max_prob = float(torch.max(probs).item())
        uncertainty = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().item())

        y_true.append(int(y.item()))
        y_pred.append(pred)
        unc.append(uncertainty)
        conf.append(max_prob)

    return ModelOutputs(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        uncertainty=np.asarray(unc, dtype=np.float64),
        confidence=np.asarray(conf, dtype=np.float64),
    )


def accuracy_coverage_curve(
    correct: np.ndarray,
    uncertainty: np.ndarray,
    coverages: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(uncertainty)  # keep least uncertain samples
    sorted_correct = correct[order]
    n = sorted_correct.shape[0]

    kept_counts = np.maximum(1, np.floor(coverages * n).astype(np.int64))
    accuracies = np.array([sorted_correct[:k].mean() for k in kept_counts], dtype=np.float64)
    risks = 1.0 - accuracies
    return {
        "coverage": coverages.copy(),
        "kept": kept_counts,
        "accuracy": accuracies,
        "risk": risks,
    }


def save_predictions_csv(path: Path, outputs: ModelOutputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "correct", "uncertainty", "confidence"])
        for yt, yp, corr, u, c in zip(
            outputs.y_true,
            outputs.y_pred,
            outputs.correct,
            outputs.uncertainty,
            outputs.confidence,
        ):
            writer.writerow([int(yt), int(yp), int(corr), float(u), float(c)])


def save_curves_csv(path: Path, curves: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "coverage", "kept", "accuracy", "risk"])
        for model_name, curve in curves.items():
            for cov, kept, acc, risk in zip(
                curve["coverage"],
                curve["kept"],
                curve["accuracy"],
                curve["risk"],
            ):
                writer.writerow([model_name, float(cov), int(kept), float(acc), float(risk)])


def plot_curves(curves: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    colors = {
        "nn": "#1f77b4",
        "laplace": "#d62728",
        "dnn2gp": "#2ca02c",
    }
    labels = {
        "nn": "Standard NN",
        "laplace": "NN + Laplace",
        "dnn2gp": "DNN2GP GP",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for key in ["nn", "laplace", "dnn2gp"]:
        curve = curves[key]
        cov_pct = 100.0 * curve["coverage"]
        ax1.plot(cov_pct, 100.0 * curve["accuracy"], marker="o", linewidth=2.1, color=colors[key], label=labels[key])
        ax2.plot(cov_pct, 100.0 * curve["risk"], marker="o", linewidth=2.1, color=colors[key], label=labels[key])

    ax1.set_title("Selective Classification: Accuracy vs Coverage")
    ax1.set_xlabel("Coverage (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    ax2.set_title("Selective Classification: Risk vs Coverage")
    ax2.set_xlabel("Coverage (%)")
    ax2.set_ylabel("Risk (%) = 100 - Accuracy")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle("MNIST rejection by uncertainty", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_test_subset(test_set, limit: int, seed: int):
    if limit <= 0 or limit >= len(test_set):
        return test_set
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_set), generator=rng)[:limit].tolist()
    return Subset(test_set, idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST selective classification with NN, Laplace, and DNN2GP.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/full_mnist_lenet_adaml2.tk")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)

    parser.add_argument("--nn-uncertainty", type=str, default="entropy", choices=["entropy", "one_minus_max"])

    parser.add_argument("--prior-prec", type=float, default=1.0)
    parser.add_argument("--laplace-cache", type=str, default="results/mnist_selective_post_prec_pp1.pt")
    parser.add_argument("--laplace-train-size", type=int, default=5000)
    parser.add_argument("--laplace-batch-size", type=int, default=64)
    parser.add_argument("--laplace-mc-samples", type=int, default=30)

    parser.add_argument("--gp-mc-samples", type=int, default=40)
    parser.add_argument("--gp-diag-cov", action="store_true", default=True)

    parser.add_argument("--test-limit", type=int, default=-1, help="-1 means full MNIST test set.")
    parser.add_argument("--coverages", type=str, default="1.0,0.95,0.90,0.80,0.70,0.50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    laplace_cache = Path(args.laplace_cache)
    if not laplace_cache.is_absolute():
        laplace_cache = Path.cwd() / laplace_cache

    coverages = parse_coverage_levels(args.coverages)
    print(f"Coverage levels: {coverages.tolist()}")

    ds = Dataset("mnist", data_folder=args.data_dir)
    test_set = make_test_subset(ds.test_set, args.test_limit, seed=args.seed + 99)
    train_loader = DataLoader(ds.train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    map_model = load_or_train_map_model(
        train_loader=train_loader,
        test_loader=eval_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=args.train_epochs,
        lr=args.train_lr,
        weight_decay=args.train_weight_decay,
        force_train=args.force_train,
    )

    model_double = copy.deepcopy(map_model).to(device=device, dtype=torch.double).eval()
    post_prec = compute_or_load_laplace_precision(
        model_double=model_double,
        train_set=ds.train_set,
        device=device,
        cache_path=laplace_cache,
        prior_prec=args.prior_prec,
        laplace_train_size=args.laplace_train_size,
        batch_size=args.laplace_batch_size,
        seed=args.seed + 1,
    )

    nn_out = predict_standard_nn(map_model, eval_loader, device=device, uncertainty_type=args.nn_uncertainty)
    lap_out = predict_laplace_mc(
        model_double=model_double,
        test_loader=eval_loader,
        post_prec=post_prec,
        device=device,
        mc_samples=args.laplace_mc_samples,
    )

    gp_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    gp_out = predict_dnn2gp_gp_sampling(
        model_double=model_double,
        test_loader=gp_loader,
        post_prec=post_prec,
        device=device,
        mc_samples=args.gp_mc_samples,
        diag_cov=args.gp_diag_cov,
    )

    if not (np.array_equal(nn_out.y_true, lap_out.y_true) and np.array_equal(nn_out.y_true, gp_out.y_true)):
        raise RuntimeError("Test labels mismatch between model outputs.")

    curves = {
        "nn": accuracy_coverage_curve(nn_out.correct, nn_out.uncertainty, coverages),
        "laplace": accuracy_coverage_curve(lap_out.correct, lap_out.uncertainty, coverages),
        "dnn2gp": accuracy_coverage_curve(gp_out.correct, gp_out.uncertainty, coverages),
    }

    save_predictions_csv(results_dir / "mnist_selective_nn_predictions.csv", nn_out)
    save_predictions_csv(results_dir / "mnist_selective_laplace_predictions.csv", lap_out)
    save_predictions_csv(results_dir / "mnist_selective_dnn2gp_predictions.csv", gp_out)
    save_curves_csv(results_dir / "mnist_selective_coverage_curves.csv", curves)
    plot_curves(curves, figures_dir / "mnist_selective_accuracy_risk_vs_coverage.png")

    print("\nFinal full-coverage accuracies:")
    print(f"  NN:      {100.0 * nn_out.correct.mean():.2f}%")
    print(f"  Laplace: {100.0 * lap_out.correct.mean():.2f}%")
    print(f"  DNN2GP:  {100.0 * gp_out.correct.mean():.2f}%")
    print("\nSaved:")
    print(f"  {results_dir / 'mnist_selective_nn_predictions.csv'}")
    print(f"  {results_dir / 'mnist_selective_laplace_predictions.csv'}")
    print(f"  {results_dir / 'mnist_selective_dnn2gp_predictions.csv'}")
    print(f"  {results_dir / 'mnist_selective_coverage_curves.csv'}")
    print(f"  {figures_dir / 'mnist_selective_accuracy_risk_vs_coverage.png'}")


if __name__ == "__main__":
    main()
