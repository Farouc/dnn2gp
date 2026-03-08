from __future__ import annotations

import numpy as np
import torch

from .compute_kernels import compute_jacobian, diagonal_function_variance, precision_to_cov_diag


def gp_predictive_distribution(
    J_test: torch.Tensor,
    J_train: torch.Tensor,
    Sigma: torch.Tensor,
    y_train: torch.Tensor | None = None,
    noise_var: float = 1e-4,
    mean_train: torch.Tensor | None = None,
    mean_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GP predictive mean and covariance using linear features."""
    if J_test.ndim != 2 or J_train.ndim != 2:
        raise ValueError("gp_predictive_distribution expects J_test and J_train with shape [N,P].")

    if Sigma.ndim == 1:
        K_tt = torch.einsum("np,p,mp->nm", J_train, Sigma, J_train)
        K_st = torch.einsum("np,p,mp->nm", J_test, Sigma, J_train)
        K_ss = torch.einsum("np,p,mp->nm", J_test, Sigma, J_test)
    elif Sigma.ndim == 2:
        K_tt = torch.einsum("np,pq,mq->nm", J_train, Sigma, J_train)
        K_st = torch.einsum("np,pq,mq->nm", J_test, Sigma, J_train)
        K_ss = torch.einsum("np,pq,mq->nm", J_test, Sigma, J_test)
    else:
        raise ValueError("Sigma must be [P] or [P,P].")

    if y_train is None:
        mean = torch.zeros(J_test.shape[0], dtype=J_test.dtype, device=J_test.device)
        if mean_test is not None:
            mean = mean + mean_test
        var = torch.diag(K_ss).clamp_min(1e-12)
        return mean, var, K_ss

    mean_train_eff = torch.zeros_like(y_train) if mean_train is None else mean_train
    mean_test_eff = torch.zeros(J_test.shape[0], dtype=J_test.dtype, device=J_test.device) if mean_test is None else mean_test
    y_centered = y_train - mean_train_eff

    eye = torch.eye(K_tt.shape[0], dtype=K_tt.dtype, device=K_tt.device)
    solve_matrix = K_tt + noise_var * eye
    alpha = torch.linalg.solve(solve_matrix, y_centered)
    pred_mean = mean_test_eff + K_st @ alpha
    pred_cov = K_ss - K_st @ torch.linalg.solve(solve_matrix, K_st.T)
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    pred_var = torch.diag(pred_cov).clamp_min(1e-12)
    return pred_mean, pred_var, pred_cov


def linearized_regression_predictive(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    precision: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mean = model(inputs).reshape(-1)
    J = compute_jacobian(model, inputs).squeeze(1)
    cov_diag = precision_to_cov_diag(precision)
    var = torch.einsum("np,p,np->n", J, cov_diag, J).clamp_min(1e-12)
    return mean, var


def linearized_logits_distribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    precision: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean logits and diagonal covariance per logit."""
    with torch.no_grad():
        mean_logits = model(inputs)
    J = compute_jacobian(model, inputs)  # [N,C,P]
    cov_diag = precision_to_cov_diag(precision)
    var_logits = torch.einsum("ncp,p,ncp->nc", J, cov_diag, J).clamp_min(1e-12)
    return mean_logits, var_logits


def gp_classification_probs_from_gaussian(
    mean_logits: torch.Tensor,
    var_logits: torch.Tensor,
    mc_samples: int = 80,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=mean_logits.device).manual_seed(seed)
    eps = torch.randn(
        (mc_samples, mean_logits.shape[0], mean_logits.shape[1]),
        generator=generator,
        device=mean_logits.device,
        dtype=mean_logits.dtype,
    )
    logits_samples = mean_logits.unsqueeze(0) + eps * torch.sqrt(var_logits).unsqueeze(0)
    probs_samples = torch.softmax(logits_samples, dim=-1)
    probs_mean = probs_samples.mean(dim=0)
    pred = torch.argmax(probs_mean, dim=1)
    confidence = probs_mean.max(dim=1).values
    entropy = -torch.sum(probs_mean * torch.log(probs_mean.clamp_min(1e-12)), dim=1)
    predictive_var = probs_samples.var(dim=0).sum(dim=1)
    return {
        "probs": probs_mean,
        "pred": pred,
        "confidence": confidence,
        "entropy": entropy,
        "predictive_var": predictive_var,
    }


def gp_regression_from_qstate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    q_state: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    mean, var = linearized_regression_predictive(model, inputs, q_state["precision"])
    return {
        "mean": mean.detach().cpu().numpy(),
        "var": var.detach().cpu().numpy(),
    }


def gp_classification_from_qstate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    q_state: dict[str, torch.Tensor],
    mc_samples: int = 80,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    mean_logits, var_logits = linearized_logits_distribution(model, inputs, q_state["precision"])
    out = gp_classification_probs_from_gaussian(mean_logits, var_logits, mc_samples=mc_samples, seed=seed)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}

