from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


result_dir = Path(
    "/home/xwx/Hand--Sam2/outputs/memory/baseline_stride80/evaluation"
)
temporal = pd.read_csv(result_dir / "temporal_metrics.csv")
summary = pd.read_csv(result_dir / "configuration_summary.csv")

# 防止视频尾部样本太少。
temporal = temporal[temporal["sequence_coverage"] >= 0.5]


def plot_temporal(strategies, filename, title):
    data = temporal[temporal["strategy"].isin(strategies)]

    fig, ax = plt.subplots(figsize=(11, 6))
    for configuration, group in data.groupby("configuration", sort=False):
        group = group.sort_values("frame_index")
        ax.plot(
            group["frame_index"],
            group["foreground_mean_iou_after"],
            label=configuration,
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Foreground IoU")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / filename, dpi=200)
    plt.close(fig)


def plot_budget(strategy, x_column, filename, title):
    data = summary[
        summary["strategy"].isin(["baseline", strategy])
    ].dropna(subset=[x_column, "foreground_mean_iou_after"])

    data = data.sort_values(x_column)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        data[x_column],
        data["foreground_mean_iou_after"],
        marker="o",
    )

    for _, row in data.iterrows():
        ax.annotate(
            row["configuration"],
            (row[x_column], row["foreground_mean_iou_after"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(x_column)
    ax.set_ylabel("Foreground IoU")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_dir / filename, dpi=200)
    plt.close(fig)


plot_temporal(
    ["baseline", "fixed"],
    "temporal_fixed.png",
    "Fixed-interval prompting",
)

plot_temporal(
    ["baseline", "adaptive"],
    "temporal_adaptive.png",
    "Adaptive correction",
)

plot_budget(
    "fixed",
    "ordinary_prompts_per_1000_hand_view_frames",
    "budget_fixed.png",
    "Fixed prompting: prompt budget vs IoU",
)

plot_budget(
    "adaptive",
    "correction_clicks_per_1000_hand_view_frames",
    "budget_adaptive.png",
    "Adaptive correction: click budget vs IoU",
)