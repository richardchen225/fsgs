"""Prepare a benchmark run from the exact configuration/checkpoint of a training run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.benchmark_io import discover_scenes


def prepare_test(args):
    from omegaconf import OmegaConf

    train_dir = Path(args.train_dir).resolve()
    with (train_dir / "final_checkpoint.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    checkpoint = Path(manifest["checkpoint"])
    if checkpoint.resolve() != train_dir / "checkpoints" / "final.ckpt" or not checkpoint.is_file():
        raise ValueError("Final checkpoint is missing or does not belong to this training run")
    cfg = OmegaConf.load(manifest["config"])
    if manifest["global_step"] != cfg.trainer.max_steps:
        raise ValueError("Final checkpoint did not reach the configured training steps")
    scenes = discover_scenes(args.bench_root, args.expected_scenes)
    context = list(cfg.dataset.dl3dv.ctx_list)
    target = list(cfg.dataset.dl3dv.tgt_list)
    if not target or len(context) <= len(target) or context[-len(target):] != target:
        raise ValueError("Test ctx_list must contain source views followed by tgt_list")
    source = context[:-len(target)]
    if source != sorted(set(source)) or len(target) != len(set(target)) or set(source) & set(target):
        raise ValueError("Source views must be ordered/unique and disjoint from unique target views")
    if any(not isinstance(i, int) or i < 0 for i in context):
        raise ValueError("Test frame indices must be non-negative integers")
    for scene_dir, scene_id in scenes:
        with (Path(scene_dir) / "transforms.json").open(encoding="utf-8") as stream:
            frames = json.load(stream)["frames"]
        if max(context) >= len(frames):
            raise ValueError(f"Test frame index out of range for scene {scene_id}")
    # Retain all architecture and preprocessing settings from the training run.
    cfg.mode = "test"
    cfg.checkpointing.load = str(checkpoint)
    cfg.checkpointing.train_pretrained_weights = None
    cfg.checkpointing.save_final = False
    cfg.dataset.dl3dv.mode = "test"
    cfg.dataset.dl3dv.roots = [str(Path(args.bench_root).resolve())]
    cfg.dataset.dl3dv.test_scan_scenes = True
    cfg.dataset.dl3dv.test_expected_scenes = args.expected_scenes
    cfg.model.encoder.mode = "test"
    cfg.model.encoder.num_test_context_views = None
    cfg.trainer.devices = args.devices
    cfg.trainer.num_nodes = 1
    cfg.data_loader.test.batch_size = 1
    cfg.test.compute_scores = True
    cfg.test.save_metrics = True
    cfg.test.output_path = str(Path(args.output_dir).resolve())
    cfg.test.save_compare = args.save_compare
    cfg.test.save_image = False
    cfg.test.save_video = False
    cfg.test.generate_video = False
    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Test checkpoint: {checkpoint}")
    print(f"Scenes: {len(scenes)}; source indices: {source}; target indices: {target}")
    print(f"Test config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("bench_root")
    preflight.add_argument("--expected-scenes", type=int, default=140)
    prepare = commands.add_parser("prepare-test")
    prepare.add_argument("train_dir")
    prepare.add_argument("bench_root")
    prepare.add_argument("output_dir")
    prepare.add_argument("config_out")
    prepare.add_argument("--devices", type=int, default=4)
    prepare.add_argument("--expected-scenes", type=int, default=140)
    prepare.add_argument("--save-compare", action="store_true")
    show = commands.add_parser("show-result")
    show.add_argument("metrics")
    show.add_argument("--expected-scenes", type=int, default=140)
    args = parser.parse_args()
    if args.command == "preflight":
        scenes = discover_scenes(args.bench_root, args.expected_scenes)
        print(f"Benchmark preflight passed: {len(scenes)} scenes, all referenced images_8 files exist")
    elif args.command == "prepare-test":
        prepare_test(args)
    else:
        with Path(args.metrics).open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary["scene_count"] != args.expected_scenes:
            raise ValueError("Benchmark result has the wrong number of unique scenes")
        print(f"Checkpoint: {summary['checkpoint']}")
        print(f"Scenes: {summary['scene_count']}")
        for key, value in summary["metrics"].items():
            print(f"{key}: {value:.6f}")
        print(f"Results: {Path(args.metrics).resolve()}")


if __name__ == "__main__":
    main()
