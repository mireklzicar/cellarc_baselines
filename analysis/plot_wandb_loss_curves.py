#!/usr/bin/env python3
"""Plot per-project loss curves from exported W&B history."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

DEFAULT_INPUT = Path("results/wandb_export/all_wandb_loss_curves.csv")
DEFAULT_OUTPUT_DIR = Path("results/wandb_export/loss_plots")
SIZE_LABELS: Dict[str, str] = {"small": "Small", "medium": "Medium", "large": "Large"}
MODE_DESCRIPTIONS: Dict[str, str] = {
    "embedding": "Models Training with Task Embeddings",
    "incontext": "In-Context Models",
    "in context": "In-Context Models",
}
MODE_SHORT_LABELS: Dict[str, str] = {
    "embedding": "Embedding",
    "incontext": "In-Context",
    "in context": "In-Context",
}
MODEL_LABELS: Dict[str, str] = {
    "tiny_recursive": "TRM",
    "transformer_act": "Transformer ACT",
    "cnn1d": "CNN1D",
    "rnn": "RNN",
    "hrm": "HRM",
    "nca1d": "NCA1D",
}
LEGEND_FONTSIZE = 16
EMBEDDING_PROJECTS = [
    "cellarc100k_50e_embedding_large",
    "cellarc100k_50e_embedding_medium",
    "cellarc100k_50e_embedding_small",
]
INCONTEXT_PROJECTS = [
    "cellarc100k_50e_incontext_large",
    "cellarc100k_50e_incontext_medium",
    "cellarc100k_50e_incontext_small",
]
OVERVIEW_FILENAME = "loss_curves_overview.png"
ROW_MODE_TITLES: Dict[str, str] = {
    "embedding": "Embedding Training",
    "incontext": "In-Context Training",
    "in context": "In-Context Training",
}
ROW_TITLE_MARGIN = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves for every project/run combination.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV produced by analysis/get_wandb_loss.py (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where project plots will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--metric",
        choices=("auto", "val", "train"),
        default="auto",
        help="Which metric to plot. 'auto' saves separate train/val figures when data is available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution when saving plots (default: 200).",
    )
    return parser.parse_args()


def simplify_run_name(run_name: str) -> str:
    """Strip size/mode suffixes (e.g., cnn1d_large_embedding -> cnn1d)."""
    if not isinstance(run_name, str) or not run_name:
        return "unknown"
    tokens = [token for token in run_name.split("_") if token]
    if len(tokens) <= 2:
        return run_name
    return "_".join(tokens[:-2])


def pretty_project_title(project: str, training_mode: Optional[str], size: Optional[str]) -> str:
    """Map raw project metadata to the requested descriptive title."""
    if isinstance(size, str) and size:
        size_key = size.lower()
        size_label = SIZE_LABELS.get(size_key, size.title())
    else:
        size_label = "Unknown"
    if isinstance(training_mode, str) and training_mode:
        mode_key = training_mode.lower()
        descriptor = MODE_DESCRIPTIONS.get(mode_key, training_mode.title())
    else:
        descriptor = "Loss Curves"
    return f"Loss Curves of {size_label} {descriptor}"


def format_model_label(base_name: str) -> str:
    """Convert internal model identifiers to display-friendly legend labels."""
    if not isinstance(base_name, str) or not base_name:
        return "Unknown"
    base = base_name.lower()
    if base in MODEL_LABELS:
        return MODEL_LABELS[base]
    cleaned = base_name.replace("_", " ")
    if cleaned.isupper():
        return cleaned
    if cleaned.islower():
        return cleaned.title()
    return cleaned


def build_color_map(model_keys: Iterable[str]) -> Dict[str, str]:
    """Assign a consistent color to each base model."""
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_len = len(palette)
    color_map: Dict[str, str] = {}
    for idx, name in enumerate(sorted({key for key in model_keys if key})):
        color_map[name] = palette[idx % palette_len]
    return color_map


def build_label_map(model_keys: Iterable[str]) -> Dict[str, str]:
    """Build display labels for each base model."""
    label_map: Dict[str, str] = {}
    for key in sorted({key for key in model_keys if key}):
        label_map[key] = format_model_label(key)
    return label_map


def row_title_label(training_mode: Optional[str], size: Optional[str]) -> str:
    """Return labels like 'In-Context\\nLarge'."""
    if isinstance(size, str) and size:
        size_label = SIZE_LABELS.get(size.lower(), size.title())
    else:
        size_label = "Unknown"
    if isinstance(training_mode, str) and training_mode:
        mode_label = MODE_SHORT_LABELS.get(training_mode.lower(), training_mode.title())
    else:
        mode_label = "Training"
    return f"{mode_label}\n{size_label}"


def plot_project_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    color_map: Dict[str, str],
    label_map: Dict[str, str],
    dpi: int,
) -> bool:
    """Plot a single metric (train or val) for all runs in a project."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drew_any = render_project_metric(ax, df, metric, color_map, label_map)
    if not drew_any:
        plt.close(fig)
        return False

    title = pretty_project_title(
        project=df["project"].iloc[0],
        training_mode=df["training_mode"].iloc[0],
        size=df["size"].iloc[0],
    )
    ax.set_title(f"{title} — {metric.title()} Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, Line2D] = {}
    for handle, lbl in zip(handles, labels):
        if not lbl or lbl == "_nolegend_" or lbl in dedup:
            continue
        dedup[lbl] = handle
    if dedup:
        legend_handles = []
        legend_labels = []
        for lbl, handle in dedup.items():
            color = handle.get_color() if hasattr(handle, "get_color") else None
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.5,
                    label=lbl,
                )
            )
            legend_labels.append(lbl)
        ax.legend(
            legend_handles,
            legend_labels,
            title="Model",
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE + 2,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def render_project_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    color_map: Dict[str, str],
    label_map: Dict[str, str],
) -> bool:
    """Render the requested metric onto an axes object."""
    column = f"{metric}_loss"
    if column not in df.columns:
        return False

    drew_any = False
    used_labels = set()

    for run_name, run_df in df.groupby("run_name"):
        series = run_df.sort_values("step")[["step", column]].dropna()
        if series.empty:
            continue
        base_key = simplify_run_name(run_name)
        label = label_map.get(base_key, format_model_label(base_key))
        color = color_map.get(base_key)
        legend_label = label if base_key not in used_labels else "_nolegend_"
        if legend_label != "_nolegend_":
            used_labels.add(base_key)
        ax.plot(
            series["step"],
            series[column],
            label=legend_label,
            color=color,
            linewidth=2.5,
        )
        drew_any = True

    return drew_any


def plot_project(
    df: pd.DataFrame,
    base_output_path: Path,
    metric_choice: str,
    dpi: int,
    color_map: Dict[str, str],
    label_map: Dict[str, str],
) -> int:
    """Create per-metric figures for a project, returning the number of figures saved."""
    metrics: Iterable[str]
    if metric_choice == "train":
        metrics = ("train",)
    elif metric_choice == "val":
        metrics = ("val",)
    else:
        metrics = ("train", "val")

    saved = 0
    for metric in metrics:
        output_path = base_output_path.with_name(f"{base_output_path.name}_{metric}.png")
        if plot_project_metric(df, metric, output_path, color_map, label_map, dpi):
            saved += 1
            print(f"Saved {output_path}")
        else:
            print(f"Skipped {df['project'].iloc[0]} ({metric}): no {metric} loss values.")
    return saved


def plot_overview_grid(
    df: pd.DataFrame,
    output_dir: Path,
    color_map: Dict[str, str],
    label_map: Dict[str, str],
    dpi: int,
) -> bool:
    """Create a combined figure with embedding projects on top and in-context on bottom."""
    project_order = EMBEDDING_PROJECTS + INCONTEXT_PROJECTS
    metrics = ("train", "val")
    nrows = len(project_order)
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), sharex=False)
    axes = axes.reshape(nrows, ncols)
    any_data = False
    row_labels: List[str] = []

    for row_idx, project_name in enumerate(project_order):
        project_df = df[df["project"] == project_name]
        if project_df.empty:
            for col in range(ncols):
                axes[row_idx, col].axis("off")
            row_labels.append("")
            continue

        row_label = row_title_label(
            training_mode=project_df["training_mode"].iloc[0],
            size=project_df["size"].iloc[0],
        )

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            plotted = render_project_metric(ax, project_df, metric, color_map, label_map)
            if plotted:
                any_data = True
                ax.set_ylabel("Loss" if col_idx == 0 else "")
                if row_idx == nrows - 1:
                    ax.set_xlabel("Step")
                else:
                    ax.set_xlabel("")
                ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_axis_off()

            if row_idx == 0:
                title = "Train Loss" if metric == "train" else "Validation Loss"
                ax.set_title(title)

        row_labels.append(row_label)

    if not any_data:
        plt.close(fig)
        return False

    handles: List[Line2D] = []
    labels = []
    for key in sorted(label_map, key=lambda k: label_map[k].lower()):
        color = color_map.get(key)
        if color is None:
            continue
        handles.append(Line2D([0], [0], color=color, label=label_map[key]))
        labels.append(label_map[key])

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=LEGEND_FONTSIZE,
            title="Model",
            title_fontsize=LEGEND_FONTSIZE + 2,
            frameon=False,
        )

    fig.tight_layout(rect=(0.12, 0.08, 1, 0.98))
    fig.canvas.draw()
    for row_idx, header in enumerate(row_labels):
        if not header:
            continue
        row_axes = axes[row_idx, :]
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y0 + ax.get_position().height for ax in row_axes)
        text_y = (y0 + y1) / 2
        fig.text(
            0.01,
            text_y,
            header,
            ha="left",
            va="center",
            fontsize=10,
        )
    output_path = output_dir / OVERVIEW_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")
    return True


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"{args.input_csv} does not exist. Run analysis/get_wandb_loss.py first.")

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise SystemExit(f"{args.input_csv} is empty.")

    model_keys = [
        simplify_run_name(name)
        for name in df["run_name"].dropna()
        if isinstance(name, str)
    ]
    color_map = build_color_map(model_keys)
    label_map = build_label_map(model_keys)

    saved = 0
    for project, project_df in df.groupby("project", sort=True):
        safe_name = project.replace("/", "_")
        base_output_path = args.output_dir / safe_name
        saved += plot_project(project_df, base_output_path, args.metric, args.dpi, color_map, label_map)

    if plot_overview_grid(df, args.output_dir, color_map, label_map, args.dpi):
        saved += 1

    if saved == 0:
        raise SystemExit("No figures were generated; check that the CSV has train/val loss values.")
    print(f"Wrote {saved} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
