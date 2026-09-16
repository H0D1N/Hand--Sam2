import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
import pandas as pd


SUMMARY_FILENAME = "configuration_summary.csv"
TEMPORAL_FILENAME = "temporal_metrics.csv"
IOU_TICK_STEP = 0.05
IOU_PADDING = 0.01


def load_results(result_dir):
    summary_path = result_dir / SUMMARY_FILENAME
    temporal_path = result_dir / TEMPORAL_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not temporal_path.is_file():
        raise FileNotFoundError(temporal_path)
    return pd.read_csv(summary_path), pd.read_csv(temporal_path)


def set_iou_axis(ax, values):
    values = pd.Series(values).dropna()
    if values.empty:
        lower, upper = 0.0, 1.0
    else:
        lower = (
            math.floor((float(values.min()) - IOU_PADDING) / IOU_TICK_STEP)
            * IOU_TICK_STEP
        )
        upper = (
            math.ceil((float(values.max()) + IOU_PADDING) / IOU_TICK_STEP)
            * IOU_TICK_STEP
        )
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        if upper <= lower:
            lower = max(0.0, lower - IOU_TICK_STEP)
            upper = min(1.0, upper + IOU_TICK_STEP)

    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(IOU_TICK_STEP))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.15, linestyle=":")


def save_figure(fig, output_dir, stem, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(
    temporal,
    strategies,
    output_dir,
    stem,
    title,
    min_sequence_coverage,
    dpi,
):
    data = temporal[
        temporal["strategy"].isin(strategies)
        & (temporal["sequence_coverage"] >= min_sequence_coverage)
    ]
    if data.empty:
        print(f"Skip {stem}: no matching temporal data")
        return

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for configuration, group in data.groupby("configuration", sort=False):
        group = group.sort_values("frame_index")
        ax.plot(
            group["frame_index"],
            group["foreground_mean_iou_after"],
            linewidth=2.2 if configuration == "baseline" else 1.4,
            label=configuration,
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Foreground IoU")
    ax.set_title(title)
    set_iou_axis(ax, data["foreground_mean_iou_after"])
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, output_dir, stem, dpi)


def budget_annotation(row, strategy):
    iou = row["foreground_mean_iou_after"]
    if row["strategy"] == "baseline":
        return f"baseline\nIoU={iou:.4f}"
    if strategy == "fixed":
        return f"N={int(row['prompt_interval'])}\nIoU={iou:.4f}"
    return f"threshold={row['iou_threshold']:.2f}\nIoU={iou:.4f}"


def plot_budget(
    summary,
    strategy,
    x_column,
    x_label,
    output_dir,
    stem,
    title,
    dpi,
):
    data = summary[summary["strategy"].isin(["baseline", strategy])].dropna(
        subset=[x_column, "foreground_mean_iou_after"]
    )
    data = data.sort_values(x_column)
    if data.empty:
        print(f"Skip {stem}: no matching summary data")
        return

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.plot(
        data[x_column],
        data["foreground_mean_iou_after"],
        marker="o",
        markersize=7,
        linewidth=2,
    )

    x_min = data[x_column].min()
    x_range = max(data[x_column].max() - x_min, 1)
    for point_index, (_, row) in enumerate(data.iterrows()):
        x = row[x_column]
        is_right = x > x_min + 0.8 * x_range
        x_offset = -7 if is_right else 7
        y_offset = 9 if strategy == "fixed" or point_index % 2 == 0 else -35
        ax.annotate(
            budget_annotation(row, strategy),
            (x, row["foreground_mean_iou_after"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8,
            ha="right" if is_right else "left",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "alpha": 0.75,
                "ec": "none",
            },
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Foreground IoU")
    ax.set_title(title)
    ax.margins(x=0.07)
    set_iou_axis(ax, data["foreground_mean_iou_after"])
    save_figure(fig, output_dir, stem, dpi)


def plot_results(result_dir, min_sequence_coverage=0.5, dpi=200):
    """从评估 CSV 生成时序和 Prompt 预算 IoU 曲线。"""
    result_dir = Path(result_dir)
    if not 0 <= min_sequence_coverage <= 1:
        raise ValueError("--min-sequence-coverage must be in [0, 1]")

    summary, temporal = load_results(result_dir)

    plot_temporal(
        temporal,
        ["baseline", "fixed"],
        result_dir,
        "temporal_fixed",
        "Fixed-interval GT-mask prompting over time",
        min_sequence_coverage,
        dpi,
    )
    plot_temporal(
        temporal,
        ["baseline", "adaptive"],
        result_dir,
        "temporal_adaptive",
        "Adaptive point correction over time",
        min_sequence_coverage,
        dpi,
    )
    plot_budget(
        summary,
        "fixed",
        "ordinary_prompts_per_1000_hand_view_frames",
        "GT-mask prompts per 1,000 hand-view frames",
        result_dir,
        "budget_fixed",
        "Fixed GT-mask prompting: prompt budget vs test IoU",
        dpi,
    )
    plot_budget(
        summary,
        "adaptive",
        "correction_clicks_per_1000_hand_view_frames",
        "Correction clicks per 1,000 hand-view frames",
        result_dir,
        "budget_adaptive",
        "Adaptive point correction: click budget vs test IoU",
        dpi,
    )
    print(f"Saved plots to: {result_dir}")
