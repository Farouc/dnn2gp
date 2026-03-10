from __future__ import annotations

import numpy as np


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def one_minus_max_prob(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probs, axis=1)


def nll_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    n, c = probs.shape
    one_hot = np.zeros((n, c), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = labels.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(conf[mask])
        ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC using Mann-Whitney statistic. labels should be 0/1."""
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.zeros_like(scores, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    auc = (np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def selective_curve(correct: np.ndarray, uncertainty: np.ndarray, coverages: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(uncertainty)  # least uncertain first
    sorted_correct = correct[order]
    n = sorted_correct.shape[0]
    keep = np.maximum(1, np.floor(coverages * n).astype(np.int64))
    acc = np.array([sorted_correct[:k].mean() for k in keep], dtype=np.float64)
    risk = 1.0 - acc
    return {
        "coverage": coverages,
        "keep": keep,
        "accuracy": acc,
        "risk": risk,
    }

