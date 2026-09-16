import pandas as pd

from inference.plotting import plot_results


def test_plot_results_writes_png_curves(tmp_path):
    pd.DataFrame([
        {
            "strategy": "baseline",
            "configuration": "baseline",
            "prompt_interval": None,
            "iou_threshold": None,
            "foreground_mean_iou_after": 0.80,
            "ordinary_prompts_per_1000_hand_view_frames": 1.0,
            "correction_clicks_per_1000_hand_view_frames": 0.0,
        },
        {
            "strategy": "fixed",
            "configuration": "fixed_interval_20",
            "prompt_interval": 20,
            "iou_threshold": None,
            "foreground_mean_iou_after": 0.90,
            "ordinary_prompts_per_1000_hand_view_frames": 50.0,
            "correction_clicks_per_1000_hand_view_frames": 0.0,
        },
        {
            "strategy": "adaptive",
            "configuration": "adaptive_iou_0p5",
            "prompt_interval": None,
            "iou_threshold": 0.5,
            "foreground_mean_iou_after": 0.92,
            "ordinary_prompts_per_1000_hand_view_frames": 1.0,
            "correction_clicks_per_1000_hand_view_frames": 30.0,
        },
    ]).to_csv(tmp_path / "configuration_summary.csv", index=False)
    pd.DataFrame([
        {
            "strategy": strategy,
            "configuration": configuration,
            "frame_index": frame_index,
            "sequence_coverage": 1.0,
            "foreground_mean_iou_after": iou,
        }
        for strategy, configuration, iou in (
            ("baseline", "baseline", 0.80),
            ("fixed", "fixed_interval_20", 0.90),
            ("adaptive", "adaptive_iou_0p5", 0.92),
        )
        for frame_index in (0, 1)
    ]).to_csv(tmp_path / "temporal_metrics.csv", index=False)

    plot_results(tmp_path, min_sequence_coverage=0.5, dpi=50)

    for stem in (
        "budget_fixed",
        "budget_adaptive",
        "temporal_fixed",
        "temporal_adaptive",
    ):
        assert (tmp_path / f"{stem}.png").is_file()
