# Inference and long-video evaluation

This package contains image prediction, full-GT long-video evaluation, and IoU
plotting. Prediction and evaluation share the same causal streaming runner;
evaluation does not call the 8-frame training validation loop.

There is only one command entry:

```bash
python3 -m inference.run_prediction {predict,evaluate} ...
```

`predict` accepts four checkpoint types: `sam2`, `framewise`, `memory`, and
`multiview`. MultiView uses an explicit PromptPlan. For example, use GT masks on
frames 0/80/160 and three correction clicks on frames 120/240:

```bash
python3 -m inference.run_prediction predict \
  --model multiview \
  --model-checkpoint outputs/multiview/experiment/checkpoints/best.pt \
  --output-dir outputs/prediction \
  --num-views 2 \
  --prompt-mode mask \
  --prompt-frame-indices 0 80 160 \
  --correction-frame-indices 120 240 \
  --correction-points 3
```

Only Prompt/correction frames need masks in `predict`; other frames require RGB
only. `--prompt-mode point` converts the GT mask on ordinary Prompt frames into
a center point. Correction frames use SAM2 positive/negative error points.

`evaluate` also accepts all four model types and requires full-frame GT. SAM2 is
evaluated with a point prompt on every frame; Framewise defaults to no prompt
(`--prompt-mode point` is optional). These two single-frame models support the
`baseline` strategy. Memory and MultiView additionally support the fixed and
adaptive sweeps below.

Three strategies are available:

- `baseline`: provide the configured prompt on frame 0 only.
- `fixed`: provide the same kind of prompt on frames `0, N, 2N, ...`.
- `adaptive`: provide the initial prompt, then measure the current prediction
  against GT before committing it. If either synchronized view of a hand is below
  the IoU threshold, run SAM2 correction for that hand and write only the corrected
  result into memory.

`--prompt-mode mask|point` controls the ordinary prompt used by `baseline` and
`fixed`. Adaptive correction always uses SAM2's positive/negative error points;
after correction is triggered, points are added until every synchronized view
reaches the configured IoU threshold. `--correction-points` sets the safety cap
on correction rounds per hand/frame (default: 10). `fixed` and `adaptive` are
evaluated as separate trajectories rather than being mixed in one run.

The default command evaluates `baseline` only. Select the other strategies and
provide multiple parameter values to generate separate curve series. For a dense
saturation curve, a useful sweep is:

```bash
--strategies baseline fixed adaptive \
--fixed-intervals 10 20 30 40 50 60 80 100 120 160 \
--adaptive-thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

This produces configurations such as `fixed_interval_40` and
`adaptive_iou_0p5`. The ordinary prompt defaults to a GT mask; pass
`--prompt-mode point` to use a GT center point instead.

Corrected frames and fixed-prompt frames are conditioning frames. Conditioning
memory is capped by `--max-condition-frames`; ordinary temporal memory follows the
model's existing spatial-memory and object-pointer windows.

Frames are loaded and encoded one timestep at a time. The evaluator retains only
`maskmem_features`, `maskmem_pos_enc`, and `obj_ptr` from committed predictions.
For adaptive evaluation, the trial output is released before the corrected/current
result is recomputed and committed. `--amp` enables CUDA bfloat16 autocast.

Memory example:

```bash
python3 -m inference.run_prediction evaluate \
  --model memory \
  --model-checkpoint outputs/memory/experiment/checkpoints/best.pt \
  --output-dir outputs/evaluation/memory \
  --dataset multiserver \
  --strategies baseline fixed adaptive \
  --fixed-intervals 20 40 80 160 \
  --adaptive-thresholds 0.3 0.5 0.7
```

Multiview example:

```bash
python3 -m inference.run_prediction evaluate \
  --model multiview \
  --model-checkpoint outputs/multiview/experiment/checkpoints/best.pt \
  --output-dir outputs/evaluation/multiview \
  --dataset multiserver \
  --num-views 2 \
  --strategies baseline fixed adaptive \
  --fixed-intervals 40 80 \
  --adaptive-thresholds 0.3 0.5
```

The command writes:

- `per_frame_metrics.csv`: every hand/view/frame prediction.
- `temporal_metrics.csv`: mean IoU at each relative frame index for every
  configuration. Videos shorter than that frame are omitted rather than padded;
  `contributing_sequences` and `sequence_coverage` expose the decreasing support.
- `configuration_summary.csv`: one aggregate row per interval or threshold.
  It also records ordinary prompt hand-views and actual correction clicks, so a
  later prompt-budget curve does not have to infer them from the interval. Both
  are additionally normalized per 1000 hand-view frames.
- `summary.json`: overall and per-sequence IoU.

Frame 0 remains in the per-frame and temporal files, while aggregate configuration
metrics exclude it by default (`--metric-start-frame 1`) because it is prompted.

## Prediction images and plotting

Add both flags to the same evaluation command when CSV, prediction PNGs, and IoU
curves are all needed:

```bash
python3 -m inference.run_prediction evaluate \
  --model memory \
  --model-checkpoint outputs/memory/experiment/checkpoints/best.pt \
  --output-dir outputs/evaluation/memory \
  --strategies baseline fixed adaptive \
  --save-predictions \
  --plot-curves
```

The script writes PNG versions of `budget_fixed`, `budget_adaptive`,
`temporal_fixed`, and `temporal_adaptive`. The IoU range is selected
automatically from the plotted values, with axis limits and major ticks rounded
to multiples of `0.05`.
