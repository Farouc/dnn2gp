"""
Snelson experiment playground.

Goal:
- Keep one file that is easy to modify for trying new approximation methods.
- Compare methods on the same train/test split.
- Save comparison plots as image files.

How to add a new method:
1) Implement a function with signature:
      (train_x, train_y, test_x, cfg) -> MethodResult
2) Add it to METHODS_TO_COMPARE (single list below).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laplace_models import NeuralNetworkRegression


@dataclass
class MethodResult:
    mean: np.ndarray
    var: np.ndarray


@dataclass
class ExperimentConfig:
    sigma_noise: float
    laplace_delta: float
    laplace_epochs: int
    laplace_hidden_size: int
    laplace_hidden_layers: int
    laplace_step_size: float
    laplace_lr_factor: float
    laplace_pred_samples: int
    gp_restarts: int
    seed: int


MethodRunner = Callable[[np.ndarray, np.ndarray, np.ndarray, ExperimentConfig], MethodResult]


@dataclass
class MethodSpec:
    name: str
    color: str
    runner: MethodRunner


def load_snelson(data_dir: Path, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_train > 200:
        raise ValueError("n_train must be <= 200 for Snelson.")

    def _read(fname: str) -> np.ndarray:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return np.array([float(v) for v in f.read().strip().splitlines()], dtype=np.float64)

    train_x = _read("train_inputs")
    train_y = _read("train_outputs")
    test_x = _read("test_inputs")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_x.shape[0])
    train_x = train_x[perm][:n_train]
    train_y = train_y[perm][:n_train]

    # Same masking as the paper script.
    mask = ((train_x < 1.5) | (train_x > 3.0)).flatten()
    train_x = train_x[mask][:, None]
    train_y = train_y[mask]
    test_x = test_x[:, None]
    return train_x, train_y, test_x


def run_gp_rbf(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, cfg: ExperimentConfig) -> MethodResult:
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(),
        alpha=cfg.sigma_noise**2,
        random_state=cfg.seed,
        n_restarts_optimizer=cfg.gp_restarts,
    )
    gp.fit(train_x, train_y)
    mean, std = gp.predict(test_x, return_std=True)
    return MethodResult(mean=mean.flatten(), var=(std**2).flatten())


def run_laplace_dnn(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, cfg: ExperimentConfig
) -> MethodResult:
    model = NeuralNetworkRegression(
        train_x,
        train_y,
        sigma_noise=cfg.sigma_noise,
        delta=cfg.laplace_delta,
        n_epochs=cfg.laplace_epochs,
        step_size=cfg.laplace_step_size,
        hidden_size=cfg.laplace_hidden_size,
        n_layers=cfg.laplace_hidden_layers + 1,
        diagonal=True,
        activation="sigmoid",
        lr_factor=cfg.laplace_lr_factor,
        seed=cfg.seed,
    )
    mean, var = model.posterior_predictive_f(
        test_x, "J", n_samples=cfg.laplace_pred_samples, compute_cov=True, diag_only=True
    )
    return MethodResult(mean=mean.flatten(), var=np.clip(var.flatten(), 1e-12, None))


def run_my_method_template(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, cfg: ExperimentConfig
) -> MethodResult:
    """
    Copy this function, implement your approximation, and add it to METHODS_TO_COMPARE.
    """
    raise NotImplementedError("Implement your custom method and add it to METHODS_TO_COMPARE.")


# =======================================================================
# EDIT ONLY THIS LIST TO CHOOSE WHAT TO COMPARE
# =======================================================================
METHODS_TO_COMPARE: list[MethodSpec] = [
    MethodSpec(name="GP-RBF", color="#4C78A8", runner=run_gp_rbf),
    MethodSpec(name="DNN-Laplace", color="#E45756", runner=run_laplace_dnn),
    # MethodSpec(name="MyMethod", color="#72B7B2", runner=run_my_method_template),
]
# =======================================================================


def plot_comparison(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    results: list[tuple[MethodSpec, MethodResult]],
    output_path: Path,
    sigma_noise: float,
    dpi: int,
):
    fig, (ax_mean, ax_var) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"hspace": 0.08}, constrained_layout=True
    )

    ax_mean.scatter(train_x[:, 0], train_y, s=8, color="black", label="train data", zorder=3)
    for spec, result in results:
        std = np.sqrt(np.clip(result.var + sigma_noise**2, 1e-12, None))
        ax_mean.plot(test_x[:, 0], result.mean, color=spec.color, linewidth=2, label=spec.name)
        ax_mean.fill_between(
            test_x[:, 0],
            result.mean - std,
            result.mean + std,
            color=spec.color,
            alpha=0.15,
        )
        ax_var.plot(test_x[:, 0], np.clip(result.var, 1e-12, None), color=spec.color, linewidth=2, label=spec.name)

    ax_mean.set_ylabel("y")
    ax_mean.set_title("Snelson: Predictive Mean ± 1 std")
    ax_mean.grid(alpha=0.25)
    ax_mean.legend(loc="upper right")

    ax_var.set_xlabel("x")
    ax_var.set_ylabel("Predictive var(f)")
    ax_var.set_title("Predictive Variance")
    ax_var.grid(alpha=0.25)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Snelson playground for method comparison.")
    parser.add_argument("--data-dir", type=str, default="data/snelson", help="Directory with Snelson files.")
    parser.add_argument("--output-dir", type=str, default="figures", help="Where to save plots.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure format.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument("--n-train", type=int, default=200, help="Number of train points (max 200).")
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument("--sigma-noise", type=float, default=0.286, help="Observation noise sigma.")
    parser.add_argument("--gp-restarts", type=int, default=10, help="RBF GP optimizer restarts.")

    parser.add_argument("--laplace-delta", type=float, default=0.1)
    parser.add_argument("--laplace-epochs", type=int, default=20000)
    parser.add_argument("--laplace-hidden-size", type=int, default=32)
    parser.add_argument("--laplace-hidden-layers", type=int, default=1)
    parser.add_argument("--laplace-step-size", type=float, default=0.1)
    parser.add_argument("--laplace-lr-factor", type=float, default=0.99)
    parser.add_argument("--laplace-pred-samples", type=int, default=1000)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast debug mode (few epochs/samples). Useful to verify wiring before full run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    laplace_epochs = args.laplace_epochs
    laplace_pred_samples = args.laplace_pred_samples
    gp_restarts = args.gp_restarts
    if args.quick:
        laplace_epochs = min(laplace_epochs, 200)
        laplace_pred_samples = min(laplace_pred_samples, 100)
        gp_restarts = min(gp_restarts, 2)

    cfg = ExperimentConfig(
        sigma_noise=args.sigma_noise,
        laplace_delta=args.laplace_delta,
        laplace_epochs=laplace_epochs,
        laplace_hidden_size=args.laplace_hidden_size,
        laplace_hidden_layers=args.laplace_hidden_layers,
        laplace_step_size=args.laplace_step_size,
        laplace_lr_factor=args.laplace_lr_factor,
        laplace_pred_samples=laplace_pred_samples,
        gp_restarts=gp_restarts,
        seed=args.seed,
    )

    train_x, train_y, test_x = load_snelson(data_dir=data_dir, n_train=args.n_train, seed=args.seed)

    if len(METHODS_TO_COMPARE) == 0:
        raise ValueError("METHODS_TO_COMPARE is empty. Add at least one method.")

    results: list[tuple[MethodSpec, MethodResult]] = []
    for spec in METHODS_TO_COMPARE:
        print(f"[RUN] {spec.name}")
        result = spec.runner(train_x, train_y, test_x, cfg)
        if result.mean.shape[0] != test_x.shape[0] or result.var.shape[0] != test_x.shape[0]:
            raise ValueError(
                f"{spec.name} returned wrong shapes: mean {result.mean.shape}, var {result.var.shape}, "
                f"expected ({test_x.shape[0]},)"
            )
        results.append((spec, result))

    out_file = output_dir / f"snelson_method_comparison.{args.format}"
    plot_comparison(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        results=results,
        output_path=out_file,
        sigma_noise=cfg.sigma_noise,
        dpi=args.dpi,
    )
    print(f"Saved comparison plot: {out_file}")


if __name__ == "__main__":
    main()
