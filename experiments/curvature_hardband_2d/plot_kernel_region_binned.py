#!/usr/bin/env python3
"""Create cleaner region-structured kernel visualizations for hardband results.

This plot is designed for paper readability:
- Samples are reordered as confident class-0 -> hard region -> confident class-1
- A binned kernel view averages local structure to remove pixel-level noise
- Region block means are annotated to show where curvature differs from NTK
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REGION_COLORS = ["#3E6DA9", "#D74E4E", "#4AA356"]
REGION_NAMES = ["confident class 0", "hard region", "confident class 1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Region-binned kernel comparison plot")
    parser.add_argument("--results-dir", type=str, default="results/curvature_hardband_2d")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins-c0", type=int, default=24)
    parser.add_argument("--bins-hard", type=int, default=16)
    parser.add_argument("--bins-c1", type=int, default=24)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def recover_unsorted_kernels(results_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    arr = np.load(results_dir / "curvature_hardband_arrays.npz")
    K_ntk_sorted = np.load(results_dir / "kernel_ntk_normalized.npy")
    K_curv_sorted = np.load(results_dir / "kernel_curvature_normalized.npy")
    summary = json.loads((results_dir / "summary_metrics.json").read_text(encoding="utf-8"))

    n = K_ntk_sorted.shape[0]
    y_first = arr["y_test"][:n]
    sigma_first = arr["sigma_test"][:n]

    # Previous script sorted only by labels; invert that permutation to recover original order.
    sort_idx_label = np.argsort(y_first, kind="stable")
    inv_idx = np.argsort(sort_idx_label)

    K_ntk = K_ntk_sorted[inv_idx][:, inv_idx]
    K_curv = K_curv_sorted[inv_idx][:, inv_idx]

    sigma_high_thr = float(summary["noise_thresholds"]["sigma_high_threshold_test"])
    return K_ntk, K_curv, y_first, sigma_first, sigma_high_thr


def build_region_order(
    y: np.ndarray,
    sigma: np.ndarray,
    sigma_high_thr: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    hard = sigma >= sigma_high_thr

    idx_c0 = np.where((~hard) & (y == 0))[0]
    idx_hard = np.where(hard)[0]
    idx_c1 = np.where((~hard) & (y == 1))[0]

    # Inside each region, sort by sigma for a smoother progression from easiest to hardest.
    idx_c0 = idx_c0[np.argsort(sigma[idx_c0])]
    idx_hard = idx_hard[np.argsort(-sigma[idx_hard])]
    idx_c1 = idx_c1[np.argsort(sigma[idx_c1])]

    order = np.concatenate([idx_c0, idx_hard, idx_c1])
    return order, [idx_c0, idx_hard, idx_c1]


def make_bins(region_indices: list[np.ndarray], n_bins: list[int]) -> list[np.ndarray]:
    bins: list[np.ndarray] = []
    for idx, b in zip(region_indices, n_bins):
        if len(idx) == 0:
            continue
        b_eff = int(max(1, min(b, len(idx))))
        # Split region into contiguous bins after sorting.
        for chunk in np.array_split(idx, b_eff):
            if len(chunk) > 0:
                bins.append(chunk)
    return bins


def binned_matrix(K: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    m = len(bins)
    out = np.zeros((m, m), dtype=np.float64)
    for i, bi in enumerate(bins):
        for j, bj in enumerate(bins):
            out[i, j] = float(np.mean(K[np.ix_(bi, bj)]))
    return out


def compute_region_block_means(K: np.ndarray, counts: list[int]) -> np.ndarray:
    c0, ch, c1 = counts
    b0 = c0
    b1 = c0 + ch
    blocks = [np.arange(0, b0), np.arange(b0, b1), np.arange(b1, K.shape[0])]
    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M[i, j] = float(np.mean(K[np.ix_(blocks[i], blocks[j])]))
    return M


def region_bin_boundaries(n_bins: list[int]) -> list[float]:
    b0 = n_bins[0]
    b1 = n_bins[0] + n_bins[1]
    return [b0 - 0.5, b1 - 0.5]


def add_region_strip(ax: plt.Axes, n_bins: list[int]) -> None:
    strip = np.concatenate(
        [
            np.zeros(n_bins[0], dtype=np.int64),
            np.ones(n_bins[1], dtype=np.int64),
            np.full(n_bins[2], 2, dtype=np.int64),
        ]
    )
    cmap = plt.matplotlib.colors.ListedColormap(REGION_COLORS)
    ax.imshow(strip[None, :], aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def annotate_region_names(fig: plt.Figure, y: float = 0.06) -> None:
    fig.text(0.22, y, REGION_NAMES[0], ha="center", fontsize=10, color=REGION_COLORS[0])
    fig.text(0.50, y, REGION_NAMES[1], ha="center", fontsize=10, color=REGION_COLORS[1])
    fig.text(0.78, y, REGION_NAMES[2], ha="center", fontsize=10, color=REGION_COLORS[2])


def add_region_background(ax: plt.Axes, bins_eff: list[int], alpha: float = 0.10) -> None:
    b0 = bins_eff[0]
    b1 = bins_eff[0] + bins_eff[1]
    regions = [(0, b0), (b0, b1), (b1, sum(bins_eff))]
    for (start, end), color in zip(regions, REGION_COLORS):
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=alpha, lw=0)
    for boundary in (b0 - 0.5, b1 - 0.5):
        ax.axvline(boundary, color="0.25", linestyle="--", lw=1.0, alpha=0.9)


def save_cross_region_affinity_figure(
    K_ntk_b: np.ndarray,
    K_curv_b: np.ndarray,
    bins_eff: list[int],
    results_dir: Path,
    dpi: int,
) -> None:
    b0 = bins_eff[0]
    b1 = bins_eff[0] + bins_eff[1]
    hard_slice = np.arange(b0, b1)
    x = np.arange(K_ntk_b.shape[0], dtype=np.float64)

    # Affinity profile: similarity of each bin to the hard-region bins.
    ntk_rows = K_ntk_b[np.ix_(hard_slice, np.arange(K_ntk_b.shape[0]))]
    curv_rows = K_curv_b[np.ix_(hard_slice, np.arange(K_curv_b.shape[0]))]
    aff_ntk = ntk_rows.mean(axis=0)
    aff_curv = curv_rows.mean(axis=0)
    std_ntk = ntk_rows.std(axis=0)
    std_curv = curv_rows.std(axis=0)
    aff_delta = aff_curv - aff_ntk

    # Region-level summary from the hard row against (C0, Hard, C1).
    reg_c0 = slice(0, b0)
    reg_hard = slice(b0, b1)
    reg_c1 = slice(b1, K_ntk_b.shape[0])
    region_ntk = np.array(
        [
            float(np.mean(aff_ntk[reg_c0])),
            float(np.mean(aff_ntk[reg_hard])),
            float(np.mean(aff_ntk[reg_c1])),
        ]
    )
    region_curv = np.array(
        [
            float(np.mean(aff_curv[reg_c0])),
            float(np.mean(aff_curv[reg_hard])),
            float(np.mean(aff_curv[reg_c1])),
        ]
    )
    region_delta = region_curv - region_ntk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7), gridspec_kw={"width_ratios": [1.85, 1.0]})

    add_region_background(ax1, bins_eff, alpha=0.09)
    ax1.plot(x, aff_ntk, color="#2E2E2E", lw=2.1, label="NTK")
    ax1.plot(x, aff_curv, color="#D74E4E", lw=2.1, label="Curvature")
    ax1.fill_between(x, aff_ntk - std_ntk, aff_ntk + std_ntk, color="#2E2E2E", alpha=0.12, lw=0)
    ax1.fill_between(x, aff_curv - std_curv, aff_curv + std_curv, color="#D74E4E", alpha=0.12, lw=0)

    ax1_t = ax1.twinx()
    ax1_t.plot(x, aff_delta, color="#1F77B4", lw=1.6, linestyle="--", label="Delta (curv - NTK)")
    ax1_t.set_ylabel("delta affinity", color="#1F77B4")
    ax1_t.tick_params(axis="y", colors="#1F77B4")

    ax1.set_title("Similarity to hard region across confidence-ordered bins")
    ax1.set_xlabel("region-ordered bins")
    ax1.set_ylabel("mean similarity to hard bins")
    ax1.grid(alpha=0.18)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)

    labels = ["C0", "Hard", "C1"]
    pos = np.arange(3, dtype=np.float64)
    w = 0.34
    ax2.bar(pos - w / 2, region_ntk, width=w, color="#2E2E2E", alpha=0.82, label="NTK")
    ax2.bar(pos + w / 2, region_curv, width=w, color="#D74E4E", alpha=0.82, label="Curvature")
    for i, d in enumerate(region_delta):
        ax2.text(i, max(region_ntk[i], region_curv[i]) + 0.015, f"{d:+.3f}", ha="center", va="bottom", fontsize=9)
    ax2.axhline(0.0, color="0.35", lw=0.8)
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("hard-region affinity (mean)")
    ax2.set_title("Region-level hard affinity")
    ax2.legend(frameon=False, loc="upper right")

    fig.suptitle("Cross-region affinity profile: NTK vs curvature", y=1.02, fontsize=13)
    fig.tight_layout()

    out_aff = results_dir / "kernel_cross_region_affinity_neurips.png"
    fig.savefig(out_aff, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_aff}")
    print(
        "Hard-affinity deltas (curv-ntk) | "
        f"C0={region_delta[0]:+.4f}, Hard={region_delta[1]:+.4f}, C1={region_delta[2]:+.4f}"
    )


def main() -> None:
    args = parse_args()
    set_style()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parents[2] / results_dir

    K_ntk_u, K_curv_u, y, sigma, sigma_thr = recover_unsorted_kernels(results_dir)
    order, region_idx = build_region_order(y, sigma, sigma_thr)

    K_ntk = K_ntk_u[order][:, order]
    K_curv = K_curv_u[order][:, order]
    K_diff = K_curv - K_ntk

    counts = [len(region_idx[0]), len(region_idx[1]), len(region_idx[2])]
    n = int(sum(counts))
    b0 = counts[0]
    b1 = counts[0] + counts[1]

    # Build binned matrices (smoother structure for paper readability).
    reg_ordered = [np.arange(0, b0), np.arange(b0, b1), np.arange(b1, n)]
    requested_bins = [args.bins_c0, args.bins_hard, args.bins_c1]
    bins = make_bins(reg_ordered, requested_bins)
    K_ntk_b = binned_matrix(K_ntk, bins)
    K_curv_b = binned_matrix(K_curv, bins)
    K_diff_b = K_curv_b - K_ntk_b

    # Actual bins per region after cap by region size.
    bins_eff = [max(1, min(requested_bins[i], counts[i])) for i in range(3)]

    block_ntk = compute_region_block_means(K_ntk, counts)
    block_curv = compute_region_block_means(K_curv, counts)
    block_delta = block_curv - block_ntk

    vmax_k = max(float(np.max(np.abs(K_ntk_b))), float(np.max(np.abs(K_curv_b))))
    vmax_d = float(np.max(np.abs(K_diff_b)))

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.08, 1.0, 0.95], hspace=0.34, wspace=0.20)

    # Region strips
    strips = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for s in strips:
        add_region_strip(s, bins_eff)

    # Row 1: binned kernels
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    im1 = ax1.imshow(K_ntk_b, cmap="magma", aspect="equal", vmin=-vmax_k, vmax=vmax_k)
    ax1.set_title("NTK kernel (region-binned)", pad=8)
    im2 = ax2.imshow(K_curv_b, cmap="magma", aspect="equal", vmin=-vmax_k, vmax=vmax_k)
    ax2.set_title("Curvature kernel (region-binned)", pad=8)
    im3 = ax3.imshow(K_diff_b, cmap="coolwarm", aspect="equal", vmin=-vmax_d, vmax=vmax_d)
    ax3.set_title("Difference (curvature - NTK)", pad=8)

    for ax in (ax1, ax2, ax3):
        for b in region_bin_boundaries(bins_eff):
            ax.axhline(b, color="white", lw=1.1, alpha=0.95)
            ax.axvline(b, color="white", lw=1.1, alpha=0.95)
        ax.set_xlabel("")
        ax.set_ylabel("region-ordered bins")

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    # Row 2: 3x3 block means with values (clear quantitative takeaway).
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])

    im4 = ax4.imshow(block_ntk, cmap="magma", aspect="equal")
    ax4.set_title("Region block means (NTK)", pad=8)
    im5 = ax5.imshow(block_curv, cmap="magma", aspect="equal")
    ax5.set_title("Region block means (Curvature)", pad=8)
    vmax_bd = float(np.max(np.abs(block_delta)))
    im6 = ax6.imshow(block_delta, cmap="coolwarm", aspect="equal", vmin=-vmax_bd, vmax=vmax_bd)
    ax6.set_title("Region block means (Curv - NTK)", pad=8)

    labels = ["C0", "Hard", "C1"]
    for ax, M, is_delta in [(ax4, block_ntk, False), (ax5, block_curv, False), (ax6, block_delta, True)]:
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(3):
                txt = f"{M[i, j]:+.3f}" if is_delta else f"{M[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.02)
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.02)

    annotate_region_names(fig, y=0.035)
    fig.suptitle("Kernel structure by confidence regime (hard-band synthetic 2D)", y=0.985, fontsize=13)

    out_path = results_dir / "kernel_comparison_region_binned_neurips.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    save_cross_region_affinity_figure(K_ntk_b, K_curv_b, bins_eff, results_dir, args.dpi)

    print(f"Saved: {out_path}")
    print(f"Counts | C0={counts[0]}, Hard={counts[1]}, C1={counts[2]}")
    print(
        "Block delta summary (Curv - NTK): "
        f"diag={[round(float(block_delta[i, i]), 4) for i in range(3)]}, "
        f"hard-vs-C0={block_delta[1,0]:+.4f}, hard-vs-C1={block_delta[1,2]:+.4f}"
    )


if __name__ == "__main__":
    main()
