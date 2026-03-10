from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def assign_parameters_from_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[idx : idx + n].view_as(p))
            idx += n


def precision_to_cov_diag(precision: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.reciprocal(precision.clamp_min(eps))


def compute_jacobian(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian wrt parameters.

    Returns a tensor of shape [N, C, P]:
    - N: number of inputs
    - C: output dimension
    - P: number of model parameters
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    jacobians = []
    for i in range(inputs.shape[0]):
        x_i = inputs[i : i + 1]
        out_i = model(x_i)
        out_i = out_i.reshape(-1)
        rows = []
        for c in range(out_i.shape[0]):
            grads = torch.autograd.grad(
                out_i[c],
                params,
                retain_graph=(c < out_i.shape[0] - 1),
                create_graph=False,
                allow_unused=False,
            )
            rows.append(torch.cat([g.reshape(-1) for g in grads], dim=0).detach())
        jacobians.append(torch.stack(rows, dim=0))
    return torch.stack(jacobians, dim=0)


def _ensure_jacobian_shape(J: torch.Tensor) -> torch.Tensor:
    if J.ndim == 2:
        return J.unsqueeze(1)
    if J.ndim == 3:
        return J
    raise ValueError("Expected Jacobian with shape [N,P] or [N,C,P].")


def compute_ntk_kernel(J: torch.Tensor) -> torch.Tensor:
    J = _ensure_jacobian_shape(J)
    return torch.einsum("ncp,mcp->nm", J, J)


def compute_vogn_kernel(J: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    J = _ensure_jacobian_shape(J)
    if Sigma.ndim == 1:
        return torch.einsum("ncp,p,mcp->nm", J, Sigma, J)
    if Sigma.ndim == 2:
        return torch.einsum("ncp,pq,mcq->nm", J, Sigma, J)
    raise ValueError("Sigma must be diagonal vector [P] or matrix [P,P].")


def compute_oggn_kernel(J: torch.Tensor, Sigma_hat: torch.Tensor) -> torch.Tensor:
    return compute_vogn_kernel(J, Sigma_hat)


def diagonal_function_variance(J: torch.Tensor, cov_diag: torch.Tensor) -> torch.Tensor:
    J = _ensure_jacobian_shape(J)
    # Sum across output channels to obtain one scalar variance per input.
    var_per_output = torch.einsum("ncp,p,ncp->nc", J, cov_diag, J)
    return var_per_output.sum(dim=1)


def save_array(path: Path, array: np.ndarray | torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(array):
        np.save(path, array.detach().cpu().numpy())
    else:
        np.save(path, array)

