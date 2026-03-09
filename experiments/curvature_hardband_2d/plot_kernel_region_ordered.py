#!/usr/bin/env python3
"""Create region-aware kernel visualizations for curvature_hardband_2d results.

Reordering strategy:
1) confident class-0 (low sigma)
2) problematic region (high sigma)
3) confident class-1 (low sigma)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot region-ordered kernel comparison.")
    parser.add_argument("--results-dir", type=str, default="results/curvature_hardband_2d")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.2,
        }
    )


def main() -> None:
    args = parse_args()
    set_style()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parents[2] / results_dir

    arr = np.load(results_dir / "curvature_hardband_arrays.npz")
    K_ntk_sorted = np.load(results_dir / "kernel_ntk_normalized.npy")
    K_curv_sorted = np.load(results_dir / "kernel_curvature_normalized.npy")
    summary = json.loads((results_dir / "summary_metrics.json").read_text(encoding="utf-8"))

    n = K_ntk_sorted.shape[0]
    y_first = arr["y_test"][:n]
    sigma_first = arr["sigma_test"][:n]

    # Recover unsorted kernels from the old label-only stable sort used in the experiment script.
    sort_idx_label = np.argsort(y_first, kind="stable")
    inv_idx = np.argsort(sort_idx_label)
    K_ntk_unsorted = K_ntk_sorted[inv_idx][:, inv_idx]
    K_curv_unsorted = K_curv_sorted[inv_idx][:, inv_idx]

    sigma_high_thr = float(summary["noise_thresholds"]["sigma_high_threshold_test"])
    hard = sigma_first >= sigma_high_thr

    region0 = np.where((~hard) & (y_first == 0))[0]  # confident class 0
    region1 = np.where(hard)[0]                      # problematic region
    region2 = np.where((~hard) & (y_first == 1))[0]  # confident class 1

    # Sort inside each region for a cleaner visual progression.
    region0 = region0[np.argsort(sigma_first[region0])]
    region1 = region1[np.argsort(-sigma_first[region1])]  # most problematic first
    region2 = region2[np.argsort(sigma_first[region2])]

    order = np.concatenate([region0, region1, region2])
    K_ntk = K_ntk_unsorted[order][:, order]
    K_curv = K_curv_unsorted[order][:, order]
    K_diff = K_curv - K_ntk

    b0 = len(region0)
    b1 = b0 + len(region1)
    boundaries = [b0 - 0.5, b1 - 0.5]

    # Region color strip for axis annotation.
    strip = np.concatenate(
        [
            np.zeros(len(region0), dtype=np.int64),
            np.ones(len(region1), dtype=np.int64),
            np.full(len(region2), 2, dtype=np.int64),
        ]
    )

    vmax_diff = float(np.max(np.abs(K_diff)))

    fig = plt.figure(figsize=(14.8, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.11, 1.0], wspace=0.25, hspace=0.06)

    region_cmap = plt.matplotlib.colors.ListedColormap(["#4C78A8", "#E45756", "#54A24B"])

    axes_main = []
    for col in range(3):
        ax_strip = fig.add_subplot(gs[0, col])
        ax_strip.imshow(strip[None, :], aspect="auto", cmap=region_cmap, interpolation="nearest")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        for spine in ax_strip.spines.values():
            spine.set_visible(False)

        ax = fig.add_subplot(gs[1, col])
        axes_main.append(ax)

    im0 = axes_main[0].imshow(K_ntk, cmap="magma", aspect="auto")
    axes_main[0].set_title("NTK kernel (region-ordered)")
    im1 = axes_main[1].imshow(K_curv, cmap="magma", aspect="auto")
    axes_main[1].set_title("Curvature kernel (region-ordered)")
    im2 = axes_main[2].imshow(K_diff, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff, aspect="auto")
    axes_main[2].set_title("Difference (curv - NTK)")

    for ax in axes_main:
        for b in boundaries:
            ax.axhline(b, color="white", lw=0.9, alpha=0.9)
            ax.axvline(b, color="white", lw=0.9, alpha=0.9)
        ax.set_xlabel("sample index")
        ax.set_ylabel("sample index")

    fig.colorbar(im0, ax=axes_main[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes_main[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes_main[2], fraction=0.046, pad=0.04)

    fig.text(0.17, 0.01, "confident class 0", ha="center", va="bottom", fontsize=10, color="#4C78A8")
    fig.text(0.50, 0.01, "problematic high-noise", ha="center", va="bottom", fontsize=10, color="#E45756")
    fig.text(0.83, 0.01, "confident class 1", ha="center", va="bottom", fontsize=10, color="#54A24B")

    out_path = results_dir / "kernel_comparison_region_ordered_neurips.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Optional compact summary: block means to quantify structural differences.
    def block_mean(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        return float(np.mean(K[np.ix_(idx_a, idx_b)]))

    reg_idx = [np.arange(0, b0), np.arange(b0, b1), np.arange(b1, n)]
    M_ntk = np.zeros((3, 3), dtype=np.float64)
    M_curv = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M_ntk[i, j] = block_mean(K_ntk, reg_idx[i], reg_idx[j])
            M_curv[i, j] = block_mean(K_curv, reg_idx[i], reg_idx[j])

    fig2, axs = plt.subplots(1, 3, figsize=(10.2, 3.5))
    ims = [
        axs[0].imshow(M_ntk, cmap="magma", aspect="equal"),
        axs[1].imshow(M_curv, cmap="magma", aspect="equal"),
        axs[2].imshow(M_curv - M_ntk, cmap="coolwarm", aspect="equal"),
    ]
    titles = ["NTK block means", "Curvature block means", "Block diff (curv-NTK)"]
    labels = ["C0", "Hard", "C1"]

    for ax, title, im in zip(axs, titles, ims):
        ax.set_title(title)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(3):
                val = (M_curv - M_ntk)[i, j] if "diff" in title.lower() else (M_curv[i, j] if "Curvature" in title else M_ntk[i, j])
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout()
    out_path2 = results_dir / "kernel_region_block_means_neurips.png"
    fig2.savefig(out_path2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path2}")
    print(
        "Region counts | "
        f"confident C0: {len(region0)}, hard: {len(region1)}, confident C1: {len(region2)}"
    )


if __name__ == "__main__":
    main()
