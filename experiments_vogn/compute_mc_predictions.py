from __future__ import annotations

import numpy as np
import torch

from .compute_kernels import assign_parameters_from_vector, flatten_parameters


def sample_weights_from_q(
    mu: torch.Tensor,
    precision: torch.Tensor,
    n_samples: int,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=mu.device).manual_seed(seed)
    eps = torch.randn((n_samples, mu.numel()), generator=generator, device=mu.device, dtype=mu.dtype)
    std = torch.rsqrt(precision.clamp_min(1e-12))
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def mc_predict(
    model: torch.nn.Module,
    q_t: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    mc_samples: int = 80,
    classification: bool = False,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    params = [p for p in model.parameters()]
    mu_model = flatten_parameters(model).detach()

    mu = q_t["mu"].to(device=mu_model.device, dtype=mu_model.dtype)
    precision = q_t["precision"].to(device=mu_model.device, dtype=mu_model.dtype)
    samples = sample_weights_from_q(mu, precision, n_samples=mc_samples, seed=seed)

    outputs = []
    with torch.no_grad():
        for w in samples:
            assign_parameters_from_vector(model, w)
            outputs.append(model(inputs).detach())
    assign_parameters_from_vector(model, mu_model)

    out = torch.stack(outputs, dim=0)  # [S,N,C] or [S,N,1]
    if not classification:
        preds = out.reshape(out.shape[0], out.shape[1])
        mean = preds.mean(dim=0)
        var = preds.var(dim=0).clamp_min(1e-12)
        return {
            "mean": mean.cpu().numpy(),
            "var": var.cpu().numpy(),
            "samples": preds.cpu().numpy(),
        }

    probs_samples = torch.softmax(out, dim=-1)
    probs_mean = probs_samples.mean(dim=0)
    pred = torch.argmax(probs_mean, dim=1)
    confidence = probs_mean.max(dim=1).values
    entropy = -torch.sum(probs_mean * torch.log(probs_mean.clamp_min(1e-12)), dim=1)
    predictive_var = probs_samples.var(dim=0).sum(dim=1)
    return {
        "probs": probs_mean.cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
        "predictive_var": predictive_var.cpu().numpy(),
    }

