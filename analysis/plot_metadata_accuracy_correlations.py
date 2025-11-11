#!/usr/bin/env python3
"""Plot correlations between per-task accuracies and CellARC metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from matplotlib.patches import Patch


COMBINED_RANK_FEATURE = "mean_rank_cov_lambda_entropy"
DEFAULT_COMBINED_FEATURES = [
    "query_window_coverage_weighted",
    "lambda",
    "avg_cell_entropy",
]

NUMERIC_FEATURES: Dict[str, str] = {
    "query_window_coverage_weighted": "Query coverage (weighted)",
    "lambda": "Langton λ",
    "avg_cell_entropy": "Avg. cell entropy",
    "avg_mutual_information_d1": "Avg. mutual information (d=1)",
    "ncd_train_query_solution": "NCD(train ↔ query+solution)",
    COMBINED_RANK_FEATURE: "Mean rank (cov ↑, λ ↓, entropy ↓)",
}

CORRELATION_LABEL = {"pearson": "r", "spearman": "ρ"}
CORRELATION_FULL_NAME = {"pearson": "Pearson r", "spearman": "Spearman ρ"}


def _format_family_label(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "Unknown"
    cleaned = str(name).strip().replace("_", " ")
    tokens = cleaned.split()
    formatted: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx].lower()
        next_token = tokens[idx + 1].lower() if idx + 1 < len(tokens) else None
        if token == "mod" and next_token in {"k", "(k)"}:
            formatted.append("mod(k)")
            idx += 2
            continue
        if token.endswith("(k)"):
            formatted.append(token)
        elif token in {"ca", "io"}:
            formatted.append(token.upper())
        else:
            formatted.append(token.capitalize())
        idx += 1
    return " ".join(formatted) or cleaned


def _family_color_map(families: List[str]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab20")
    colors = cmap.colors if hasattr(cmap, "colors") else [cmap(i) for i in np.linspace(0, 1, cmap.N)]
    return {family: colors[idx % len(colors)] for idx, family in enumerate(families)}


def _download_metadata(repo_id: str, split: str) -> Path:
    """Download the JSONL file for the requested split from the HF Hub."""
    filename = f"data/{split}.jsonl"
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
    )


def _load_metadata(repo_id: str, splits: Iterable[str]) -> pd.DataFrame:
    """Load per-episode metadata for the requested splits."""
    rows: List[Dict[str, object]] = []
    for split in splits:
        jsonl_path = _download_metadata(repo_id, split)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                payload = json.loads(line)
                meta = payload.get("meta") or {}
                episode_id = payload.get("id") or meta.get("fingerprint")
                if episode_id is None:
                    raise ValueError(f"Missing episode id in {jsonl_path} line {line_idx}")
                record: Dict[str, object] = {
                    "episode_id": episode_id,
                    "split": split,
                    "family": meta.get("family"),
                }
                for key in NUMERIC_FEATURES:
                    record[key] = meta.get(key)
                rows.append(record)
    return pd.DataFrame(rows)


def _add_combined_rank_metric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df[COMBINED_RANK_FEATURE] = np.nan
        return df
    coverage_rank = df["query_window_coverage_weighted"].rank(method="average", ascending=True, pct=True)
    lambda_rank = df["lambda"].rank(method="average", ascending=False, pct=True)
    entropy_rank = df["avg_cell_entropy"].rank(method="average", ascending=False, pct=True)
    df[COMBINED_RANK_FEATURE] = (coverage_rank + lambda_rank + entropy_rank) / 3.0
    return df


def _compute_correlation(x: pd.Series, y: pd.Series, method: str) -> Tuple[float, int]:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    return float(x[mask].corr(y[mask], method=method)), int(mask.sum())


def _plot_numeric_correlations(
    df: pd.DataFrame,
    models: List[str],
    splits: List[str],
    output_path: Path,
    corr_type: str,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    split_colors = {split: colors[idx % len(colors)] for idx, split in enumerate(splits)}

    feature_items = list(NUMERIC_FEATURES.items())
    fig, axes = plt.subplots(
        len(feature_items),
        len(models),
        figsize=(4.5 * len(models), 3.3 * len(feature_items)),
        squeeze=False,
    )

    corr_symbol = CORRELATION_LABEL[corr_type]

    for row_idx, (feature_key, feature_label) in enumerate(feature_items):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            for split in splits:
                subset = df[df["split"] == split]
                ax.scatter(
                    subset[feature_key],
                    subset[model],
                    label=split if row_idx == 0 and col_idx == 0 else None,
                    s=30,
                    alpha=0.8,
                    color=split_colors[split],
                    edgecolors="none",
                )
            corr, count = _compute_correlation(df[feature_key], df[model], corr_type)
            if count >= 2 and corr == corr:
                ax.text(
                    0.02,
                    0.95,
                    f"{corr_symbol} = {corr:.2f}\nn = {count}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
                fit_mask = df[feature_key].notna() & df[model].notna()
                xs = df.loc[fit_mask, feature_key]
                ys = df.loc[fit_mask, model]
                if xs.nunique() > 1:
                    slope, intercept = np.polyfit(xs, ys, 1)
                    span = np.linspace(xs.min(), xs.max(), 50)
                    ax.plot(span, slope * span + intercept, color="#444444", linewidth=1, alpha=0.6)

            if row_idx == len(feature_items) - 1:
                ax.set_xlabel(feature_label)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(f"{model} accuracy")
            else:
                ax.set_ylabel("")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    axes[0][0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Metadata vs per-task accuracy ({CORRELATION_FULL_NAME[corr_type]})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _plot_family_distributions(
    df: pd.DataFrame,
    models: List[str],
    output_path: Path,
    target_split: Optional[str] = None,
) -> None:
    if target_split:
        split_df = df[df["split"] == target_split].copy()
        if split_df.empty:
            print(f"[family plot] No rows found for split '{target_split}', skipping plot.")
            return
        active_split_label = target_split.replace("_", " ")
    else:
        split_df = df.copy()
        active_split_label = "all splits"

    split_df = split_df.dropna(subset=["family"])
    if split_df.empty:
        print("[family plot] No family metadata available, skipping plot.")
        return

    families = sorted(split_df["family"].unique(), key=_format_family_label)
    family_labels = {family: _format_family_label(family) for family in families}
    family_colors = _family_color_map(families)

    fig_width = 5.4 + max(0, len(models) - 1) * 3.8 + 1.4
    fig_height = 4.2
    rc_overrides = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 130,
    }
    with plt.rc_context(rc_overrides):
        fig, axes = plt.subplots(1, len(models), figsize=(fig_width, fig_height), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, model in zip(axes, models):
            model_values: List[np.ndarray] = []
            model_families: List[str] = []
            for fam in families:
                values = split_df.loc[split_df["family"] == fam, model].dropna().to_numpy()
                if values.size == 0:
                    continue
                model_values.append(values)
                model_families.append(fam)
            if not model_values:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(
                model_values,
                widths=0.48,
                patch_artist=True,
                showfliers=False,
                medianprops={"linewidth": 1.4, "color": "#1c1c1c"},
                whiskerprops={"linewidth": 1.0, "color": "#555555"},
                capprops={"linewidth": 1.0, "color": "#555555"},
            )
            for box, fam in zip(bp["boxes"], model_families):
                color = family_colors[fam]
                box.set(facecolor=color, edgecolor=color, alpha=0.9)
            for whisker in bp["whiskers"]:
                whisker.set_alpha(0.8)
            for cap in bp["caps"]:
                cap.set_alpha(0.8)
            ax.set_xticks([])
            ax.set_xlim(0.4, len(model_families) + 0.6)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.25)
            if ax is axes[0]:
                ax.set_ylabel("Accuracy")
            ax.set_title(model)
        legend_handles = [
            Patch(facecolor=family_colors[fam], edgecolor=family_colors[fam], label=family_labels[fam]) for fam in families
        ]
        legend_cols = 1
        last_ax = axes[-1]
        last_ax.legend(
            legend_handles,
            [family_labels[fam] for fam in families],
            loc="center left",
            bbox_to_anchor=(0.96, 0.85),
            ncol=legend_cols,
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.14, wspace=0.08)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=250)
        plt.close(fig)


def _write_correlation_table(df: pd.DataFrame, models: List[str], output_path: Path, corr_type: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature_key, feature_label in NUMERIC_FEATURES.items():
        for model in models:
            corr, count = _compute_correlation(df[feature_key], df[model], corr_type)
            rows.append(
                {
                    "correlation_type": corr_type,
                    "metric": feature_key,
                    "metric_label": feature_label,
                    "model": model,
                    "correlation": corr,
                    "n": count,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_path, index=False)
    return corr_df


def _compute_split_correlations(df: pd.DataFrame, models: List[str], splits: List[str], corr_type: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in splits:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        for feature_key, feature_label in NUMERIC_FEATURES.items():
            for model in models:
                corr, count = _compute_correlation(subset[feature_key], subset[model], corr_type)
                rows.append(
                    {
                        "correlation_type": corr_type,
                        "split": split,
                        "model": model,
                        "metric": feature_key,
                        "metric_label": feature_label,
                        "correlation": corr,
                        "n": count,
                    }
                )
    return pd.DataFrame(rows)


def _plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    models: List[str],
    splits: List[str],
    output_path: Path,
    corr_type: str,
) -> None:
    if corr_df.empty:
        return

    row_labels = [f"{model} / {split}" for model in models for split in splits]
    corr_df = corr_df.copy()
    corr_df["row_label"] = corr_df["model"] + " / " + corr_df["split"]
    columns_order = list(NUMERIC_FEATURES.keys())
    pivot = corr_df.pivot_table(index="row_label", columns="metric", values="correlation")
    pivot = pivot.reindex(index=row_labels).reindex(columns=columns_order)

    fig, ax = plt.subplots(
        figsize=(1.8 * len(columns_order) + 2, 0.7 * len(row_labels) + 2),
        constrained_layout=False,
    )
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(columns_order)))
    ax.set_xticklabels([NUMERIC_FEATURES[col] for col in columns_order], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i, row in enumerate(row_labels):
        for j, col in enumerate(columns_order):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                text = "–"
                text_color = "black"
            else:
                percent = value * 100
                text = f"{percent:.0f}%"
                text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(CORRELATION_FULL_NAME[corr_type])
    ax.set_title(f"Per-split accuracy vs metadata correlations ({CORRELATION_FULL_NAME[corr_type]})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _plot_combined_heatmap(
    corr_df: pd.DataFrame,
    models: List[str],
    output_path: Path,
    corr_type: str,
    features: List[str],
) -> None:
    if corr_df.empty:
        return
    columns_order = features
    pivot = corr_df.pivot_table(index="model", columns="metric", values="correlation")
    pivot = pivot.reindex(index=models).reindex(columns=columns_order)

    fig, ax = plt.subplots(
        figsize=(1.8 * len(columns_order) + 2, 0.7 * len(models) + 1.5),
        constrained_layout=False,
    )
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns_order)))
    ax.set_xticklabels([NUMERIC_FEATURES[col] for col in columns_order], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i, model in enumerate(models):
        for j, col in enumerate(columns_order):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                text = "–"
                text_color = "black"
            else:
                percent = value * 100
                text = f"{percent:.0f}%"
                text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(CORRELATION_FULL_NAME[corr_type])
    ax.set_title(f"Combined splits: accuracy vs metadata ({CORRELATION_FULL_NAME[corr_type]})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _plot_combined_scatter_grid(
    df: pd.DataFrame,
    models: List[str],
    features: List[str],
    output_path: Path,
    corr_types: List[str],
    splits: List[str],
) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(
        len(features),
        len(models),
        figsize=(4.5 * len(models), 3.0 * len(features)),
        squeeze=False,
    )

    corr_labels = {ctype: CORRELATION_LABEL.get(ctype, ctype) for ctype in corr_types}
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    split_colors = {split: colors[idx % len(colors)] for idx, split in enumerate(splits)}

    for row_idx, feature in enumerate(features):
        feature_label = NUMERIC_FEATURES.get(feature, feature)
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            for split in splits:
                subset = df[df["split"] == split]
                if subset.empty:
                    continue
                ax.scatter(
                    subset[feature],
                    subset[model],
                    s=25,
                    alpha=0.75,
                    color=split_colors[split],
                    edgecolors="none",
                    label=split if (row_idx == 0 and col_idx == 0) else None,
                )
            text_lines = []
            n_value = None
            for corr_type in corr_types:
                corr, count = _compute_correlation(df[feature], df[model], corr_type)
                label = corr_labels[corr_type]
                if count >= 2 and corr == corr:
                    text_lines.append(f"{label} = {corr:.2f}")
                    if n_value is None:
                        n_value = count
                else:
                    text_lines.append(f"{label} = –")
            if n_value is not None:
                text_lines.append(f"n = {n_value}")
            if text_lines:
                ax.text(
                    0.02,
                    0.95,
                    "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            mask = df[feature].notna() & df[model].notna()
            xs = df.loc[mask, feature]
            ys = df.loc[mask, model]
            if xs.nunique() > 1:
                slope, intercept = np.polyfit(xs, ys, 1)
                span = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(span, slope * span + intercept, color="#444444", linewidth=1, alpha=0.6)
            ax.set_xlabel(feature_label)
            if col_idx == 0:
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(model)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    legend_splits = [split for split in splits if split in split_colors]
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", label=split, markerfacecolor=split_colors[split], markersize=6)
        for split in legend_splits
    ]
    if handles:
        fig.legend(handles, legend_splits, loc="upper right", fontsize=9, frameon=False)

    corr_titles = ", ".join(CORRELATION_FULL_NAME.get(ct, ct) for ct in corr_types)
    fig.suptitle(f"Combined splits: metadata vs accuracy ({corr_titles})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metadata vs per-task accuracy correlations.")
    parser.add_argument(
        "--per-task-csv",
        type=Path,
        default=Path("outputs/analysis/de_bruijn_vs_gpt_per_task_accuracy.csv"),
        help="CSV produced by export_per_task_accuracy.py with per-episode accuracies.",
    )
    parser.add_argument(
        "--metadata-repo",
        default="mireklzicar/cellarc_100k_meta",
        help="HF dataset repo containing metadata JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_interpolation_100", "test_extrapolation_100"],
        help="Splits to consider (must exist both in the csv and HF dataset).",
    )
    parser.add_argument(
        "--family-plot-split",
        default="test_interpolation_100",
        help="Restrict the CA family plot to this split (use 'all' to include every split).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory where plots and CSVs will be stored.",
    )
    parser.add_argument(
        "--combined-subdir",
        type=Path,
        default=Path("combined"),
        help="Subdirectory (inside output-dir) for combined-split artifacts.",
    )
    parser.add_argument(
        "--combined-features",
        nargs="+",
        default=DEFAULT_COMBINED_FEATURES,
        help="Metadata features to include in combined heatmaps/scatters.",
    )
    parser.add_argument(
        "--correlation-types",
        nargs="+",
        choices=("pearson", "spearman"),
        default=["pearson"],
        help="Correlation types to compute (default: pearson).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    per_task_df = pd.read_csv(args.per_task_csv)
    per_task_df = per_task_df[per_task_df["split"].isin(args.splits)]
    if per_task_df.empty:
        raise SystemExit("No rows in the per-task CSV match the requested splits.")

    model_columns = [col for col in per_task_df.columns if col not in {"episode_id", "split"}]
    if not model_columns:
        raise SystemExit("No model accuracy columns found in the per-task CSV.")

    metadata_df = _load_metadata(args.metadata_repo, args.splits)
    merged = pd.merge(per_task_df, metadata_df, on=["episode_id", "split"], how="inner")
    if len(merged) != len(per_task_df):
        missing = per_task_df.merge(merged[["episode_id", "split"]], on=["episode_id", "split"], how="left", indicator=True)
        missing_ids = missing[missing["_merge"] == "left_only"]["episode_id"].tolist()
        raise SystemExit(f"Missing metadata for {len(missing_ids)} episodes: {missing_ids[:5]}")
    merged = _add_combined_rank_metric(merged)
    combined_dir = (args.output_dir / args.combined_subdir).resolve()
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_features = [feat for feat in args.combined_features if feat in NUMERIC_FEATURES]
    if not combined_features:
        raise SystemExit("No valid combined features specified.")

    family_plot_split: Optional[str]
    if args.family_plot_split:
        normalized = args.family_plot_split.strip().lower()
        if normalized in {"", "all", "none"}:
            family_plot_split = None
        else:
            family_plot_split = args.family_plot_split.strip()
    else:
        family_plot_split = None

    for corr_type in args.correlation_types:
        suffix = ""
        if len(args.correlation_types) > 1 or corr_type != "pearson":
            suffix = f"_{corr_type}"

        numeric_plot = args.output_dir / f"metadata_vs_accuracy{suffix}.png"
        family_plot = args.output_dir / f"ca_family_vs_accuracy{suffix}.png"
        corr_csv = args.output_dir / f"metadata_accuracy_correlations{suffix}.csv"
        split_corr_csv = args.output_dir / f"metadata_accuracy_correlations_by_split{suffix}.csv"
        heatmap_png = args.output_dir / f"metadata_accuracy_heatmap{suffix}.png"
        heatmap_combined_png = combined_dir / f"metadata_accuracy_heatmap_combined{suffix}.png"
        combined_corr_csv = combined_dir / f"metadata_accuracy_correlations_combined{suffix}.csv"

        _plot_numeric_correlations(merged, model_columns, args.splits, numeric_plot, corr_type)
        _plot_family_distributions(merged, model_columns, family_plot, target_split=family_plot_split)
        overall_corr = _write_correlation_table(merged, model_columns, corr_csv, corr_type)
        split_corr = _compute_split_correlations(merged, model_columns, args.splits, corr_type)
        split_corr.to_csv(split_corr_csv, index=False)
        _plot_correlation_heatmap(split_corr, model_columns, args.splits, heatmap_png, corr_type)
        overall_filtered = overall_corr[overall_corr["metric"].isin(combined_features)].copy()
        overall_filtered.to_csv(combined_corr_csv, index=False)
        _plot_combined_heatmap(overall_filtered, model_columns, heatmap_combined_png, corr_type, combined_features)

        print(f"[{corr_type}] Wrote numeric scatter plot to {numeric_plot}")
        print(f"[{corr_type}] Wrote CA family distribution plot to {family_plot}")
        print(f"[{corr_type}] Wrote correlation table to {corr_csv}")
        print(f"[{corr_type}] Wrote per-split correlation table to {split_corr_csv}")
        print(f"[{corr_type}] Wrote correlation heatmap to {heatmap_png}")
        print(f"[{corr_type}] Wrote combined-splits correlation CSV to {combined_corr_csv}")
        print(f"[{corr_type}] Wrote combined-splits heatmap to {heatmap_combined_png}")

    scatter_suffix = ""
    if len(args.correlation_types) == 1:
        only = args.correlation_types[0]
        scatter_suffix = "" if only == "pearson" else f"_{only}"
    else:
        scatter_suffix = "_" + "-".join(args.correlation_types)
    combined_scatter_png = combined_dir / f"metadata_accuracy_scatter_combined{scatter_suffix}.png"
    _plot_combined_scatter_grid(
        merged,
        model_columns,
        combined_features,
        combined_scatter_png,
        args.correlation_types,
        args.splits,
    )
    print(f"[{'+'.join(args.correlation_types)}] Wrote combined-splits scatter grid to {combined_scatter_png}")


if __name__ == "__main__":
    main()
