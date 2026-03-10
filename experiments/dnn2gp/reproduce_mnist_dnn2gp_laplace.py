#!/usr/bin/env python3
"""
Reproduce DNN2GP on MNIST and visualize:
1) GP kernel covariance matrix
2) Laplace posterior mean (class probabilities)

The 300 examples are grouped by class (30 per class by default).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn2gp import compute_laplace
from src.neural_networks import LeNet5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def load_or_train_model(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, float]:
    model = LeNet5(input_channels=1, dims=28, num_classes=10).to(device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not args.force_train:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            test_acc = float(state.get("test_accuracy", 0.0))
        else:
            model.load_state_dict(state)
            test_acc = 0.0
        if test_acc <= 0.0:
            test_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Loaded checkpoint: {checkpoint_path} | test_acc={100 * test_acc:.2f}%")
        return model, test_acc

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:02d}/{args.max_epochs} | "
            f"train_loss={train_loss:.4f} | test_acc={100 * test_acc:.2f}%"
        )
        if test_acc >= args.target_acc:
            print(f"Target test accuracy reached: {100 * test_acc:.2f}%")
            break

    final_acc = evaluate_accuracy(model, test_loader, device)
    torch.save({"model_state": model.state_dict(), "test_accuracy": final_acc}, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path} | test_acc={100 * final_acc:.2f}%")
    return model, final_acc


def build_balanced_subset(
    mnist_test_set,
    n_per_class: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(mnist_test_set, "data") or not hasattr(mnist_test_set, "targets"):
        raise RuntimeError("MNIST test set needs `data` and `targets` attributes.")

    data = torch.as_tensor(mnist_test_set.data)
    targets = torch.as_tensor(mnist_test_set.targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    selected_indices = []
    for cls in range(num_classes):
        cls_idx = torch.where(targets == cls)[0]
        if cls_idx.numel() < n_per_class:
            raise ValueError(
                f"Class {cls} has only {cls_idx.numel()} samples, expected at least {n_per_class}."
            )
        perm = torch.randperm(cls_idx.numel(), generator=generator)
        selected_indices.append(cls_idx[perm[:n_per_class]])

    selected_indices = torch.cat(selected_indices, dim=0)
    selected_targets = targets[selected_indices]
    selected_images = data[selected_indices].unsqueeze(1).to(torch.double) / 255.0

    sort_idx = torch.argsort(selected_targets, stable=True)
    return selected_images[sort_idx], selected_targets[sort_idx]


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def assign_parameters_from_vector(model: nn.Module, vector: torch.Tensor) -> None:
    index = 0
    with torch.no_grad():
        for param in model.parameters():
            n = param.numel()
            param.copy_(vector[index:index + n].view_as(param))
            index += n


def laplace_posterior_mean_probs(
    model: nn.Module,
    x: torch.Tensor,
    post_prec: torch.Tensor,
    mc_samples: int,
    seed: int,
) -> torch.Tensor:
    theta_map = flatten_parameters(model)
    post_prec = post_prec.to(device=theta_map.device, dtype=theta_map.dtype)
    std = torch.rsqrt(post_prec.clamp_min(1e-12))
    generator = torch.Generator(device=theta_map.device).manual_seed(seed)

    was_training = model.training
    model.eval()

    mean_probs = None
    with torch.no_grad():
        for _ in tqdm(range(mc_samples), desc="Laplace posterior mean (MC)"):
            eps = torch.randn(
                theta_map.shape,
                generator=generator,
                device=theta_map.device,
                dtype=theta_map.dtype,
            )
            theta_sample = theta_map + std * eps
            assign_parameters_from_vector(model, theta_sample)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            mean_probs = probs if mean_probs is None else (mean_probs + probs)

        assign_parameters_from_vector(model, theta_map)

    if was_training:
        model.train()

    return (mean_probs / mc_samples).cpu()


def gradient_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).flatten())
        else:
            grads.append(param.grad.detach().flatten())
    return torch.cat(grads)


def compute_kernel_sum_memory_efficient(
    model: nn.Module,
    x: torch.Tensor,
) -> np.ndarray:
    """
    Memory-efficient equivalent of compute_kernel(Jacobians, agg_type='sum').

    For agg_type='sum':
      K_ij = sum_{k,l,p} J_{i,k,p} J_{j,l,p}
           = <sum_k J_{i,k,:}, sum_l J_{j,l,:}>
    So we only store one summed Jacobian vector per sample.
    """
    was_training = model.training
    model.eval()
    summed_jacobians = []

    for i in tqdm(range(x.shape[0]), desc="Summed Jacobians"):
        xi = x[i:i + 1]
        logits = model(xi)
        grad_sum = None
        for cls in range(logits.shape[1]):
            retain = cls < logits.shape[1] - 1
            logits[0, cls].backward(retain_graph=retain)
            g = gradient_vector(model)
            grad_sum = g if grad_sum is None else (grad_sum + g)
            model.zero_grad(set_to_none=True)
        summed_jacobians.append(grad_sum.cpu().float())

    if was_training:
        model.train()

    J_sum = torch.stack(summed_jacobians, dim=0)
    kernel = J_sum @ J_sum.T
    return kernel.numpy()


def class_boundaries(labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=num_classes)
    boundaries = np.cumsum(counts)[:-1]
    return counts, boundaries


def plot_results(
    kernel_sorted: np.ndarray,
    posterior_mean_sorted: np.ndarray,
    labels_sorted: np.ndarray,
    test_acc: float,
    output_path: Path,
    num_classes: int = 10,
) -> None:
    _, boundaries = class_boundaries(labels_sorted, num_classes=num_classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(kernel_sorted, cmap="coolwarm")
    axes[0].set_title("GP kernel matrix (DNN2GP)")
    axes[0].set_xlabel("Examples (sorted by class)")
    axes[0].set_ylabel("Examples (sorted by class)")
    for b in boundaries:
        axes[0].axhline(b - 0.5, color="black", linewidth=0.6)
        axes[0].axvline(b - 0.5, color="black", linewidth=0.6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        posterior_mean_sorted,
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"Laplace posterior mean (test acc: {100 * test_acc:.2f}%)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Examples (sorted by class)")
    axes[1].set_xticks(np.arange(num_classes))
    for b in boundaries:
        axes[1].axhline(b - 0.5, color="white", linewidth=0.6)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce DNN2GP Laplace figure on MNIST.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="results/artifacts/lenet5_mnist.pt")
    parser.add_argument("--output", type=str, default="results/artifacts/mnist_dnn2gp_laplace.png")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--target-acc", type=float, default=0.99)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--subset-per-class", type=int, default=30, help="30 -> 300 examples total.")
    parser.add_argument("--prior-prec", type=float, default=1.0)
    parser.add_argument("--laplace-batch-size", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=30)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    req = device_arg.lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if req == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")

    if req == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        print("Requested `--device cuda`, but CUDA is unavailable. Falling back automatically.")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")

    if req == "mps":
        if mps_ok:
            return torch.device("mps")
        print("Requested `--device mps`, but MPS is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{device_arg}'. Use one of: auto, cpu, cuda, mps.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_device = resolve_device(args.device)
    print(f"Training device: {train_device}")

    try:
        from src.datasets import Dataset
    except ModuleNotFoundError as exc:
        if exc.name == "torchvision":
            raise SystemExit(
                "Missing dependency: torchvision. Install it first, e.g. `pip install torchvision`."
            ) from exc
        raise

    dataset = Dataset("mnist", data_folder=args.data_dir)
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(train_device.type == "cuda"),
    )
    test_loader = DataLoader(
        dataset.test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(train_device.type == "cuda"),
    )

    model, test_acc = load_or_train_model(args, train_loader, test_loader, train_device)
    print(f"Final test accuracy: {100 * test_acc:.2f}%")

    x_subset, y_subset = build_balanced_subset(
        dataset.test_set,
        n_per_class=args.subset_per_class,
        num_classes=10,
        seed=args.seed,
    )
    print(f"Subset shape: {x_subset.shape}, labels shape: {y_subset.shape}")

    # The Laplace + Jacobian pipeline below uses float64 heavily.
    # MPS does not support float64 end-to-end, so switch to CPU for this stage.
    laplace_device = train_device
    if laplace_device.type == "mps":
        print("Switching to CPU for Laplace/kernel stage because float64 is required.")
        laplace_device = torch.device("cpu")

    model = model.to(device=laplace_device, dtype=torch.double)
    x_subset_device = x_subset.to(device=laplace_device, dtype=torch.double)

    laplace_dataset = TensorDataset(x_subset, y_subset.to(torch.double))
    laplace_loader = DataLoader(
        laplace_dataset,
        batch_size=args.laplace_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Computing Laplace diagonal posterior precision...")
    post_prec = compute_laplace(
        model=model,
        train_loader=laplace_loader,
        prior_prec=args.prior_prec,
        device=laplace_device,
    )

    print("Computing Laplace posterior mean...")
    posterior_mean = laplace_posterior_mean_probs(
        model=model,
        x=x_subset_device,
        post_prec=post_prec,
        mc_samples=args.mc_samples,
        seed=args.seed + 1,
    )

    print("Computing DNN2GP kernel matrix...")
    kernel = compute_kernel_sum_memory_efficient(model=model, x=x_subset_device)

    output_path = Path(args.output)
    plot_results(
        kernel_sorted=kernel,
        posterior_mean_sorted=posterior_mean.numpy(),
        labels_sorted=y_subset.numpy(),
        test_acc=test_acc,
        output_path=output_path,
        num_classes=10,
    )
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
