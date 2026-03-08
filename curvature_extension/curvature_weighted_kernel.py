"""Minimal helpers for a curvature-weighted DNN2GP kernel extension.

This module intentionally reuses the repository's existing Laplace + dual GP
objects and only adds small helper functions.
"""

from __future__ import annotations

import numpy as np


def compute_dnn2gp_kernel(jacobian_features: np.ndarray, prior_cov: np.ndarray) -> np.ndarray:
    """Original DNN2GP kernel K = J S0 J^T."""
    return jacobian_features @ prior_cov @ jacobian_features.T


def compute_curvature_weighted_features(
    jacobian_features: np.ndarray,
    lambda_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute phi(x) = sqrt(Lambda) * J(x) for scalar-output regression.

    For the Gaussian regression setup used in this repo, lambda_values are
    effectively constant (1/sigma_noise^2). We still compute the scalar from
    data for an explicit check.
    """
    lambda_scalar = float(np.mean(lambda_values))
    phi = np.sqrt(lambda_scalar) * jacobian_features
    return phi, lambda_scalar


def compute_curvature_weighted_kernel(
    jacobian_features: np.ndarray,
    prior_cov: np.ndarray,
    lambda_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Curvature-weighted kernel K_curv = phi S0 phi^T."""
    phi, lambda_scalar = compute_curvature_weighted_features(jacobian_features, lambda_values)
    return phi @ prior_cov @ phi.T, lambda_scalar
