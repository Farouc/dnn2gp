from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.datasets import Dataset
from src.dnn2gp import compute_laplace
from src.neural_networks import LeNet5


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
        print("Requested CUDA but it is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError("Unsupported device. Use one of: auto, cuda, cpu.")


def load_mnist_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = LeNet5(input_channels=1, dims=28, num_classes=10).to(device=device, dtype=torch.double)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_mnist_data(data_dir: str):
    ds = Dataset("mnist", data_folder=data_dir)
    return ds.train_set, ds.test_set


def sample_balanced_test_subset(test_set, n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.as_tensor(test_set.targets, dtype=torch.long)
    images = torch.as_tensor(test_set.data).unsqueeze(1).to(torch.double) / 255.0
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for cls in range(10):
        idx = torch.where(targets == cls)[0]
        if idx.numel() < n_per_class:
            raise ValueError(f"Class {cls} has only {idx.numel()} samples; need {n_per_class}.")
        perm = torch.randperm(idx.numel(), generator=generator)
        selected.append(idx[perm[:n_per_class]])
    selected = torch.cat(selected, dim=0)
    order = torch.argsort(targets[selected], stable=True)
    selected = selected[order]
    return images[selected].clone(), targets[selected].clone()


def sample_digit_pairs(
    test_set,
    digit_a: int,
    digit_b: int,
    n_pairs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.as_tensor(test_set.targets, dtype=torch.long)
    images = torch.as_tensor(test_set.data).unsqueeze(1).to(torch.double) / 255.0
    generator = torch.Generator().manual_seed(seed)

    idx_a = torch.where(targets == digit_a)[0]
    idx_b = torch.where(targets == digit_b)[0]
    if idx_a.numel() < n_pairs or idx_b.numel() < n_pairs:
        raise ValueError("Not enough images for requested number of interpolation pairs.")
    pick_a = idx_a[torch.randperm(idx_a.numel(), generator=generator)[:n_pairs]]
    pick_b = idx_b[torch.randperm(idx_b.numel(), generator=generator)[:n_pairs]]
    return images[pick_a].clone(), images[pick_b].clone()


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads, dim=0)


def compute_or_load_post_prec(
    model: torch.nn.Module,
    train_set,
    device: torch.device,
    cache_path: Path,
    prior_prec: float,
    train_subset_size: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached Laplace posterior precision: {cache_path}")
        return torch.load(cache_path, map_location=device).to(device=device, dtype=torch.double)

    targets = torch.as_tensor(train_set.targets, dtype=torch.long)
    images = torch.as_tensor(train_set.data).unsqueeze(1).to(torch.double) / 255.0
    generator = torch.Generator().manual_seed(seed)
    n_total = images.shape[0]
    subset = min(train_subset_size, n_total)
    perm = torch.randperm(n_total, generator=generator)[:subset]

    x_train = images[perm]
    y_train = targets[perm].to(torch.double)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        "Computing Laplace posterior precision on MNIST train subset "
        f"(n={subset}, batch_size={batch_size}, prior_prec={prior_prec})"
    )
    post_prec = compute_laplace(model=model, train_loader=loader, prior_prec=prior_prec, device=device)
    post_prec = post_prec.detach().to(device=device, dtype=torch.double)
    torch.save(post_prec.detach().cpu(), cache_path)
    print(f"Saved Laplace posterior precision cache: {cache_path}")
    return post_prec


def compute_uncertainty_from_post_prec(
    model: torch.nn.Module,
    images: torch.Tensor,
    post_prec: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    inv_post_prec = torch.reciprocal(post_prec.to(device=device, dtype=torch.double).clamp_min(1e-12))
    probs, epi, alea, entropy, max_prob, pred = [], [], [], [], [], []

    for i in tqdm(range(images.shape[0]), desc="DNN2GP uncertainties"):
        x = images[i : i + 1].to(device=device, dtype=torch.double)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        p = torch.softmax(logits[0], dim=0)
        lam = torch.diag(p) - torch.outer(p, p)

        jac_rows = []
        for cls in range(logits.shape[1]):
            retain_graph = cls < logits.shape[1] - 1
            logits[0, cls].backward(retain_graph=retain_graph)
            jac_rows.append(_gradient_vector(model))
            model.zero_grad(set_to_none=True)
        jac = torch.stack(jac_rows, dim=0)

        cov_f = torch.einsum("kp,p,mp->km", jac, inv_post_prec, jac)
        var_f = torch.diag(lam @ cov_f @ lam).clamp_min(0.0)
        noise = (p - p**2).clamp_min(0.0)

        probs.append(p.detach().cpu().numpy())
        epi.append(var_f.detach().cpu().numpy())
        alea.append(noise.detach().cpu().numpy())
        entropy.append(float((-p * torch.log(p.clamp_min(1e-12))).sum().item()))
        max_prob.append(float(p.max().item()))
        pred.append(int(torch.argmax(p).item()))

    return {
        "probs": np.stack(probs, axis=0),
        "epistemic": np.stack(epi, axis=0),
        "aleatoric": np.stack(alea, axis=0),
        "entropy": np.asarray(entropy),
        "max_prob": np.asarray(max_prob),
        "pred": np.asarray(pred),
    }


def image_strip(images: torch.Tensor, max_items: int = 10, pad: int = 1) -> np.ndarray:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected images shape [N,1,H,W].")
    n = min(max_items, images.shape[0])
    tiles = [images[i, 0].detach().cpu().numpy() for i in range(n)]
    h, w = tiles[0].shape
    canvas = np.zeros((h, n * w + (n - 1) * pad), dtype=np.float64)
    for i, tile in enumerate(tiles):
        start = i * (w + pad)
        canvas[:, start : start + w] = tile
    return canvas
