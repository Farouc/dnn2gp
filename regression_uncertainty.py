import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dual_models import DualGPRegression, DualLinearRegression
from laplace_models import NeuralNetworkRegression
from neural_networks import SimpleMLP
from vogn import VOGN

torch.set_default_dtype(torch.double)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Snelson regression uncertainty experiments.")
    parser.add_argument("--data-dir", type=str, default="data/snelson", help="Directory containing Snelson text files.")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory where figures are written.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure format.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI (for raster formats).")
    parser.add_argument("--n-train", type=int, default=200, help="Number of Snelson training points to load (max 200).")
    parser.add_argument("--laplace-epochs", type=int, default=20000, help="Training epochs for Laplace model.")
    parser.add_argument("--vi-epochs", type=int, default=16000, help="Training epochs for VI model.")
    parser.add_argument("--laplace-pred-samples", type=int, default=1000, help="MC samples for Laplace predictive.")
    parser.add_argument("--vi-pred-samples", type=int, default=1000, help="MC samples for VI predictive.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for VI training.")
    parser.add_argument("--gp-restarts", type=int, default=20, help="Restarts for sklearn GP hyperparameter optimization.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument("--usetex", action="store_true", help="Enable LaTeX text rendering in Matplotlib.")
    return parser.parse_args()


def setup_plot_style(usetex=False):
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("seaborn-white")
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.style"] = "normal"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc("text", usetex=usetex)


def get_palette():
    try:
        import brewer2mpl

        bmap = brewer2mpl.get_map("Set1", "qualitative", 4)
        return list(bmap.mpl_colors)
    except Exception:
        return sns.color_palette("Set1", 4)


class ToyData1D:
    def __init__(self, train_x, train_y, test_x=None, x_min=None, x_max=None, n_test=None, dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        if test_x is not None:
            self.test_x = np.array(test_x, dtype=dtype)[:, None]
            self.n_test = self.test_x.shape[0]
        else:
            self.n_test = n_test
            self.test_x = np.linspace(x_min, x_max, num=n_test, dtype=dtype)[:, None]


def load_snelson_data(data_dir, n=200, dtype=np.float64):
    if n > 200:
        raise ValueError("Only 200 data points are available for Snelson.")

    def _load_snelson(filename):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")], dtype=dtype)

    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)


def plot_uncertainties(
    X,
    y,
    X_test,
    means,
    variances,
    labels,
    colors,
    output_path,
    skip=None,
    sigma_noise=None,
    plot_on_one_figure=True,
    dpi=220,
):
    if skip is None:
        skip = []
    extra_var = sigma_noise**2 if sigma_noise is not None else 0.0
    if plot_on_one_figure:
        plt.figure(figsize=(7, 4.0))

    for i, (label_i, mean_i, var_i, color_i) in enumerate(zip(labels, means, variances, colors)):
        if i in skip:
            continue
        if not plot_on_one_figure:
            plt.figure(figsize=(7, 4.0))
        plt.scatter(X[:, 0], y, s=2, color="black")
        plt.plot(X_test[:, 0], mean_i, label=label_i, color=color_i)
        plt.fill_between(
            X_test[:, 0],
            mean_i - np.sqrt(var_i + extra_var),
            mean_i + np.sqrt(var_i + extra_var),
            alpha=0.2,
            color=color_i,
        )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.ylim([-2.7, 2])
        plt.legend(loc=(0.5, -0.02))
        plt.xlim([-4, 10])
        plt.tight_layout()
        if not plot_on_one_figure:
            plt.savefig(output_path, dpi=dpi)
            plt.close()
    if plot_on_one_figure:
        plt.savefig(output_path, dpi=dpi)
        plt.close()


def compute_dual_quantities(model, sigma_noise, X, y=None):
    grads = []
    lam = np.zeros((X.shape[0],))
    residual = np.zeros((X.shape[0],))
    if y is None:
        y = np.zeros((X.shape[0],))
    for i, (xi, yi) in enumerate(zip(X, y)):
        model.zero_grad()
        output = model(xi)
        output.backward()
        grad = torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()
        grads.append(grad)
        output_value = float(output.detach().cpu().numpy().squeeze())
        if torch.is_tensor(yi):
            yi_value = float(yi.detach().cpu().item())
        else:
            yi_value = float(np.asarray(yi).squeeze())
        residual[i] = sigma_noise ** (-2) * (output_value - yi_value)
        lam[i] = sigma_noise ** (-2)

    jacobian = torch.stack(grads).detach().numpy()
    return jacobian, residual, lam


def compute_dual_quantities_mc(opt, model, sigma_noise, Xs, ys, mc_samples=1):
    parameters = opt.param_groups[0]["params"]
    jacobians = [[[], [], []] for _ in range(len(Xs))]
    precision = opt.state["precision"]
    mu = opt.state["mu"]
    for _ in range(mc_samples):
        raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
        p = torch.addcdiv(mu, raw_noise, torch.sqrt(precision), value=1.0)
        vector_to_parameters(p, parameters)
        for i, (X, y) in enumerate(zip(Xs, ys)):
            jacobian, res, lam = compute_dual_quantities(model, sigma_noise, X, y)
            jacobians[i][0].append(jacobian)
            jacobians[i][1].append(res)
            jacobians[i][2].append(lam)
    vector_to_parameters(mu, parameters)
    return [[np.concatenate(jacobians[i][j], 0) for j in range(len(jacobians[i]))] for i in range(len(Xs))]


def main():
    args = parse_args()
    setup_plot_style(usetex=args.usetex)
    colors = get_palette()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snelson_data = load_snelson_data(data_dir=data_dir, n=args.n_train)
    X_snelson, y_snelson = snelson_data.train_x, snelson_data.train_y.reshape((-1,))
    X_test_snelson = np.linspace(-4, 10, 1000).reshape((-1, 1))

    mask = ((X_snelson < 1.5) | (X_snelson > 3)).flatten()
    X = X_snelson[mask, :]
    y = y_snelson[mask]
    X_test = X_test_snelson

    fig_data = output_dir / f"snelson_data.{args.format}"
    plt.figure(figsize=(7, 4.0))
    plt.scatter(X, y, s=2, color="k", label="data")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Snelson data")
    plt.tight_layout()
    plt.savefig(fig_data, dpi=args.dpi)
    plt.close()

    hidden_size = 32
    hidden_layers = 1
    input_size = 1
    activation = "sigmoid"
    delta = 0.1
    sigma_noise = 0.286
    initial_lr = 0.1

    primal_nn = NeuralNetworkRegression(
        X,
        y,
        sigma_noise=sigma_noise,
        delta=delta,
        n_epochs=args.laplace_epochs,
        step_size=initial_lr,
        hidden_size=hidden_size,
        n_layers=hidden_layers + 1,
        diagonal=True,
        activation=activation,
        lr_factor=0.99,
    )

    y_test_pred = primal_nn.predictive_map(X_test)
    nn_mc_mean, nn_mc_var = primal_nn.posterior_predictive_f(
        X_test, "J", n_samples=args.laplace_pred_samples, compute_cov=True, diag_only=True
    )

    m_0, S_0 = np.zeros(primal_nn.d), 1 / delta * np.eye(primal_nn.d)
    (Us, Ss), vs = primal_nn.UsSs("J"), primal_nn.vs("J")
    X_hat, y_hat, s_noise = Us, Us @ primal_nn.theta_star - vs / Ss, 1 / np.sqrt(Ss)

    X_hat_test, _ = primal_nn.UsSs("J", X=X_test, y=np.ones((X_test.shape[0],)))
    dual_gp = DualGPRegression(X_hat, y_hat, s_noise, m_0, S_0=S_0)
    _, gp_var_primal = dual_gp.posterior_predictive_f(X_hat_test, diag_only=True)

    dual_blr = DualLinearRegression(X_hat, y_hat, s_noise, m_0, S_0=S_0)
    dual_blr.P_post = np.diag(np.diag(dual_blr.P_post))
    dual_blr.S_post = np.diag(1 / np.diag(dual_blr.P_post))
    _, blr_var = dual_blr.posterior_predictive_f(X_hat_test, diag_only=True)

    gp_rbf = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(),
        n_restarts_optimizer=args.gp_restarts,
        random_state=args.seed,
        alpha=sigma_noise**2,
    ).fit(X, y)

    gp_rbf_pred_mean, gp_rbf_pred_std = gp_rbf.predict(X_test, return_std=True)
    gp_rbf_pred_var = gp_rbf_pred_std**2

    labels = ["DNN-Laplace", "DNN2GP-Laplace", "GP-RBF", "DNN2GP-LaplaceDiag"]
    means = [nn_mc_mean, y_test_pred, gp_rbf_pred_mean, y_test_pred]
    variances = [nn_mc_var, gp_var_primal, gp_rbf_pred_var, blr_var]
    laplace_path = output_dir / f"regression_uncertainty_laplace.{args.format}"
    plot_uncertainties(
        X,
        y,
        X_test,
        means,
        variances,
        labels,
        colors,
        output_path=laplace_path,
        skip=[3],
        sigma_noise=sigma_noise,
        dpi=args.dpi,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    delta = 1.0
    objective = lambda pred, target: -torch.distributions.Normal(target, sigma_noise).log_prob(pred).sum()
    model_vi = SimpleMLP(input_size, hidden_size, hidden_layers + 1, activation)

    optimizer_vi = VOGN(
        model_vi,
        train_set_size=X.shape[0],
        prior_prec=delta,
        lr=1,
        betas=(0.9, 0.995),
        num_samples=10,
        inital_prec=50.0,
    )

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    X_test_t = torch.from_numpy(X_test)

    for t in range(args.vi_epochs):
        for xb, yb in DataLoader(TensorDataset(Xt, yt), batch_size=args.batch_size):
            def closure():
                optimizer_vi.zero_grad()
                output = model_vi(xb).flatten()
                loss = objective(output, yb)
                return loss, output, None

            loss, _ = optimizer_vi.step(closure)

        if (t + 1) % 500 == 0:
            logits = model_vi(Xt).flatten()
            log_loss = objective(logits, yt).detach().item()
            print(f"Epoch {t + 1}, Log-loss: {log_loss:.6f}")

    vi_pred_mc = torch.stack(
        optimizer_vi.get_mc_predictions(model_vi.forward, X_test_t, mc_samples=args.vi_pred_samples),
        0,
    )
    vi_pred_mc_mean = vi_pred_mc.mean(0).detach().numpy().flatten()
    vi_pred_mc_var = vi_pred_mc.var(0).detach().numpy().flatten()

    mc_samples = 1
    torch.manual_seed(args.seed)
    [X_vi_hat, residual, lam], [X_test_vi_hat, _, _] = compute_dual_quantities_mc(
        optimizer_vi,
        model_vi,
        sigma_noise,
        [Xt, X_test_t],
        [y, None],
        mc_samples=mc_samples,
    )

    s_noise_vi = np.sqrt(mc_samples) * np.power(lam, -0.5)
    y_vi_hat = X_vi_hat @ model_vi.weights.detach().numpy() - residual / lam
    dual_gp_vi = DualGPRegression(X_vi_hat, y_vi_hat, s_noise_vi, m_0, S_0=S_0)
    _, gp_var_vi = dual_gp_vi.posterior_predictive_f(X_test_vi_hat, diag_only=True)

    dual_blr_vi = DualLinearRegression(X_vi_hat, y_vi_hat, s_noise_vi, m_0, S_0=S_0)
    dual_blr_vi.m_post = optimizer_vi.state["mu"].detach().numpy()
    dual_blr_vi.P_post = np.diag(optimizer_vi.state["precision"].detach().numpy())
    dual_blr_vi.S_post = np.diag(1 / np.diag(dual_blr_vi.P_post))
    _, blr_var_vi = dual_blr_vi.posterior_predictive_f(X_test_vi_hat, diag_only=True)

    labels = ["DNN-VI", "DNN2GP-VI", "GP-RBF", "DNN2GP-VI-diag"]
    means = [vi_pred_mc_mean, vi_pred_mc_mean, gp_rbf_pred_mean, vi_pred_mc_mean]
    variances = [vi_pred_mc_var, gp_var_vi, gp_rbf_pred_var, blr_var_vi]
    vi_path = output_dir / f"regression_uncertainty_vi.{args.format}"
    plot_uncertainties(
        X,
        y,
        X_test,
        means,
        variances,
        labels,
        colors,
        output_path=vi_path,
        skip=[3],
        sigma_noise=sigma_noise,
        dpi=args.dpi,
    )

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
