# Train and Test on DL3DV-140

Run from the repository root in the existing training environment:

```bash
bash scripts/train_then_bench.sh /data/dl3dv /data/dl3dv-140
```

The first path is the training dataset, including `train_index.json` and the
validation `test_index.json`. The second path needs no index:

```text
dl3dv-140/
  <scene-id>/nerfstudio/transforms.json
  <scene-id>/nerfstudio/images_8/...
```

The launcher checks all scene images before training. Missing/invalid scenes cause
an error, rather than being skipped or replaced. The default expected count is 140.

Defaults are 4 training GPUs, 4 test GPUs, and 20,000 optimizer steps. Customize:

```bash
GPU_NUM=4 TEST_GPU_NUM=4 TRAIN_STEPS=20000 SAVE_COMPARE=1 \
  bash scripts/train_then_bench.sh /data/dl3dv /data/dl3dv-140 \
  checkpointing.train_pretrained_weights=weights/pre_wm.safetensors
```

Additional arguments are Hydra training overrides. Model/GIR switches, view lists,
and preprocessing settings carry over from the saved training configuration.
The launcher explicitly sets the dataset roots, device counts, `mode`, run paths,
and `checkpointing.load`. This launcher starts a new training run, not a resume.
`RUN_DIR` may name a new output directory; an existing directory is rejected.

Training must reach `max_steps` successfully. All ranks then participate in saving
`train/checkpoints/final.ckpt`; rank zero publishes `train/final_checkpoint.json`.
Testing uses exactly this checkpoint, never a file selected by modification time.
Optimizer and scheduler state remain in the checkpoint; loss-network weights are
omitted by the existing checkpoint hook.

## Evaluation Protocol

This uses the current project's view lists on the DL3DV-140 scene set. It does not
implement a separate official benchmark view-selection protocol. With the current
experiment configuration, source indices are `[1, 5, 10, 15, 20]`, and targets are
`[2, 3, 7, 12, 17, 18]` (zero-based indices into `transforms.json` frame order).
`ctx_list` must be the source list followed by `tgt_list`. Frame ordering and image
preprocessing follow the existing test loader. Actual render dimensions and frame
indices are recorded per scene, so comparisons can use the same settings.

Test uses batch size 1 per GPU. Standard distributed sampling is retained;
padding duplicates are de-duplicated by scene before computing global metrics.
The aggregate is the mean of per-scene target-view means, matching the existing
per-scene metric definition. Both the final Lightning metrics and saved results
use this de-duplicated aggregate.

## Output

```text
output/train_bench_<timestamp>_<pid>/
  train.log
  train/.hydra/config.yaml
  train/checkpoints/final.ckpt
  train/final_checkpoint.json
  bench_config/config.yaml
  test.log
  bench_dl3dv140/metrics.json
  bench_dl3dv140/per_scene.csv
  result.txt
```

`metrics.json` includes PSNR/SSIM/LPIPS, scene counts, padding duplicate count, GIR
statistics, checkpoint path, and the resolved test configuration. `per_scene.csv`
includes individual metrics, source/target indices, image shape, and GIR statistics.
The final terminal output is also saved in `result.txt`. Comparison images are
optional (`SAVE_COMPARE=1`) and live under the benchmark output directory.

To rerun only testing after a completed training run:

```bash
python -m src.main --config-path /absolute/path/to/run/bench_config --config-name config \
  hydra.run.dir=/absolute/path/to/run/bench_rerun
```

For a deliberately smaller smoke test dataset, set `EXPECTED_SCENES` to its count;
results record that count and must not be reported as a complete DL3DV-140 run.
