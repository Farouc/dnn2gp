import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

cmap = plt.get_cmap("tab10")
tab10 = lambda c: rgb2hex(cmap(c))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot kernel and predictive arrays and save figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing .npy result arrays.")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory where figures are written.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure format.")
    parser.add_argument("--prefix", type=str, default="CIFAR_Laplace", help="Array prefix, e.g. CIFAR_Laplace.")
    parser.add_argument("--kernel-path", type=str, default=None, help="Override path to kernel .npy.")
    parser.add_argument("--gp-mean-path", type=str, default=None, help="Override path to gp predictive mean .npy.")
    parser.add_argument("--pred-mean-path", type=str, default=None, help="Override path to predictive mean .npy.")
    parser.add_argument("--var-f-path", type=str, default=None, help="Override path to predictive latent variance .npy.")
    parser.add_argument("--var-y-path", type=str, default=None, help="Override path to predictive observation noise .npy.")
    parser.add_argument("--labels-path", type=str, default=None, help="Optional .npy file containing per-row labels.")
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Comma-separated class labels for x-axis (defaults to 0..K-1).",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively in addition to saving.")
    parser.add_argument("--use-gl", action="store_true", help="Use WebGL heatmap traces.")
    return parser.parse_args()


def resolve_path(override, results_dir, prefix, suffix):
    if override:
        return Path(override)
    return results_dir / f"{prefix}{suffix}"


def infer_prefix_from_kernel(results_dir, requested_prefix):
    requested = results_dir / f"{requested_prefix}_kernel.npy"
    if requested.exists():
        return requested_prefix

    kernels = sorted(results_dir.glob("*_kernel.npy"))
    if len(kernels) == 1:
        return kernels[0].name[: -len("_kernel.npy")]
    if len(kernels) == 0:
        raise FileNotFoundError(
            f"No *_kernel.npy found in {results_dir}. Run kernel generation first or pass --kernel-path."
        )
    names = ", ".join(k.name for k in kernels)
    raise FileNotFoundError(
        f"Prefix '{requested_prefix}' not found in {results_dir}. Available kernel files: {names}. "
        "Pass --prefix or --kernel-path."
    )


def make_color_fn(labels):
    uniques = list(dict.fromkeys(labels.tolist()))
    color_map = {label: tab10(i % 10) for i, label in enumerate(uniques)}
    return lambda c: color_map[c]


def build_default_labels(n_rows, n_classes):
    repeats = int(np.ceil(n_rows / n_classes))
    return np.repeat(np.arange(n_classes), repeats)[:n_rows]


def save_figure(fig, output_path, show=False):
    if show:
        fig.show()
    pio.write_image(fig, str(output_path))


def save_kernel_fallback(kernel, output_path):
    vmax = max(-kernel.min(), kernel.max())
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(kernel, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto", origin="upper")
    ax.set_xlabel("data examples")
    ax.set_ylabel("data examples")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_observations_fallback(values, classes, output_path, plot_var=False):
    vmax = values.max() if plot_var else 1.0
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    im = ax.imshow(values, cmap="Reds", vmin=0.0, vmax=vmax, aspect="auto", origin="upper")
    ax.set_xlabel("class")
    ax.set_ylabel("data examples")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_kernel(kernel, labels, label_to_color, output_path, use_gl=False, show=False):
    n_rows = len(labels)
    zmax = max(-kernel.min(), kernel.max())
    heatmap = go.Heatmapgl if use_gl else go.Heatmap
    data = [
        heatmap(
            x=np.arange(1, n_rows + 1),
            y=np.arange(1, n_rows + 1),
            z=kernel,
            zmin=-zmax,
            zmax=zmax,
            colorscale=[[0.0, "blue"], [0.5, "white"], [1.0, "red"]],
            colorbar=dict(thickness=15, tickformat="e", exponentformat="e", showexponent="none"),
        )
    ]
    shapes = []
    jump = 1 / n_rows
    for idx, label in enumerate(labels):
        pos = idx / n_rows
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": pos,
                "y0": 1.01,
                "x1": pos + jump,
                "y1": 1.01,
                "line": {"color": label_to_color(label), "width": 10},
            }
        )
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "yanchor": -1,
                "x0": -0.01,
                "y0": 1 - pos,
                "x1": -0.01,
                "y1": 1 - (pos + jump),
                "line": {"color": label_to_color(label), "width": 10},
            }
        )

    layout = go.Layout(
        font=dict(size=24.5),
        autosize=False,
        showlegend=False,
        width=655,
        height=595,
        xaxis=dict(title="data examples", autorange=True, zeroline=False, linecolor="black", mirror=True),
        yaxis=dict(
            title="data examples",
            autorange="reversed",
            ticklen=14,
            zeroline=False,
            mirror=True,
            linecolor="black",
        ),
        margin=go.layout.Margin(l=98, r=5, b=70, t=10, pad=0),
        shapes=shapes,
    )
    fig = go.Figure(data=data, layout=layout)
    try:
        save_figure(fig, output_path, show=show)
    except Exception:
        save_kernel_fallback(kernel, output_path)


def plot_observations(
    values,
    labels,
    classes,
    label_to_color,
    output_path,
    use_gl=False,
    width=300,
    plot_var=False,
    show=False,
):
    n_rows, n_classes = values.shape
    zmin = 0
    zmax = values.max() if plot_var else 1
    heatmap = go.Heatmapgl if use_gl else go.Heatmap
    data = [
        heatmap(
            x=classes,
            y=np.arange(1, n_rows + 1),
            z=values,
            zmin=zmin,
            zmax=zmax,
            colorscale=[[0.0, "white"], [1.0, "red"]],
            colorbar=dict(thickness=15, exponentformat="e", showexponent="none"),
        )
    ]
    shapes = []
    jump = 1 / n_rows
    for idx, label in enumerate(labels):
        pos = idx / n_rows
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "yanchor": -1,
                "x0": -0.02,
                "y0": 1 - pos,
                "x1": -0.02,
                "y1": 1 - (pos + jump),
                "line": {"color": label_to_color(label), "width": 10},
            }
        )
    layout = go.Layout(
        font=dict(size=21.5),
        autosize=False,
        showlegend=False,
        width=width,
        height=595,
        xaxis=dict(
            title="class",
            type="category",
            autorange=True,
            zeroline=False,
            linecolor="black",
            nticks=n_classes,
            mirror=True,
        ),
        yaxis=dict(
            title="data examples",
            autorange="reversed",
            zeroline=False,
            mirror=True,
            ticklen=12,
            linecolor="black",
        ),
        margin=go.layout.Margin(l=95, r=5, b=70, t=10, pad=0),
        shapes=shapes,
    )
    fig = go.Figure(data=data, layout=layout)
    try:
        save_figure(fig, output_path, show=show)
    except Exception:
        save_observations_fallback(values, classes, output_path, plot_var=plot_var)


def maybe_load(path):
    if path.exists():
        return np.load(path)
    return None


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = infer_prefix_from_kernel(results_dir, args.prefix)

    kernel_path = resolve_path(args.kernel_path, results_dir, prefix, "_kernel.npy")
    gp_mean_path = resolve_path(args.gp_mean_path, results_dir, prefix, "_gp_predictive_mean.npy")
    pred_mean_path = resolve_path(args.pred_mean_path, results_dir, prefix, "_predictive_mean.npy")
    var_f_path = resolve_path(args.var_f_path, results_dir, prefix, "_predictive_var_f.npy")
    var_y_path = resolve_path(args.var_y_path, results_dir, prefix, "_predictive_noise.npy")

    kernel = np.load(kernel_path)
    gp_mean = maybe_load(gp_mean_path)
    pred_mean = maybe_load(pred_mean_path)
    var_f = maybe_load(var_f_path)
    var_y = maybe_load(var_y_path)

    n_rows = kernel.shape[0]
    reference_obs = pred_mean if pred_mean is not None else gp_mean
    if reference_obs is None:
        raise FileNotFoundError(
            "Could not find either predictive_mean or gp_predictive_mean arrays. "
            "Pass --pred-mean-path or --gp-mean-path."
        )

    n_classes = reference_obs.shape[1]
    if args.class_labels is None:
        classes = np.arange(n_classes)
    else:
        classes = [c.strip() for c in args.class_labels.split(",")]
        if len(classes) != n_classes:
            raise ValueError(f"--class-labels has {len(classes)} labels but expected {n_classes}.")

    if args.labels_path:
        labels = np.load(Path(args.labels_path))
        if len(labels) != n_rows:
            raise ValueError(f"labels length {len(labels)} does not match kernel size {n_rows}.")
    else:
        labels = build_default_labels(n_rows, n_classes)
    label_to_color = make_color_fn(labels)

    kernel_out = output_dir / f"{prefix}_kernel.{args.format}"
    plot_kernel(kernel, labels, label_to_color, kernel_out, use_gl=args.use_gl, show=args.show)

    if pred_mean is not None:
        pred_out = output_dir / f"{prefix}_pred_mean_ste.{args.format}"
        plot_observations(
            pred_mean,
            labels,
            classes,
            label_to_color,
            pred_out,
            use_gl=args.use_gl,
            width=550,
            show=args.show,
        )

    if gp_mean is not None:
        gp_out = output_dir / f"{prefix}_gp_pred_mean.{args.format}"
        plot_observations(
            gp_mean,
            labels,
            classes,
            label_to_color,
            gp_out,
            use_gl=args.use_gl,
            width=550,
            show=args.show,
        )

    if var_f is not None:
        var_f_out = output_dir / f"{prefix}_var_f.{args.format}"
        plot_observations(
            var_f,
            labels,
            classes,
            label_to_color,
            var_f_out,
            use_gl=args.use_gl,
            width=450,
            plot_var=True,
            show=args.show,
        )

    if var_y is not None:
        var_y_out = output_dir / f"{prefix}_var_y.{args.format}"
        plot_observations(
            var_y,
            labels,
            classes,
            label_to_color,
            var_y_out,
            use_gl=args.use_gl,
            width=450,
            plot_var=True,
            show=args.show,
        )

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
