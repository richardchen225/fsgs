import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.evaluation.benchmark_io import (
    discover_scenes, image_8_path, summarize_scenes, write_json, write_results,
)
from scripts.benchmark_workflow import prepare_test


def create_scene(root, name, frame_count=3):
    directory = root / name / "nerfstudio"
    (directory / "images_8").mkdir(parents=True)
    frames = []
    for index in range(frame_count):
        filename = f"frame_{index:05}.png"
        (directory / "images_8" / filename).touch()
        frames.append({"file_path": f"images/{filename}"})
    write_json(directory / "transforms.json", {"frames": frames})
    return directory


def row(scene, psnr):
    return {
        "scene": scene, "source_indices": [0, 2], "target_indices": [1],
        "image_shape": [518, 518], "psnr_ours": psnr,
        "ssim_ours": 0.6, "lpips_ours": 0.3, "gir": {"map_gaussians": 100.0},
    }


class BenchmarkIOTests(unittest.TestCase):
    def test_scan_without_index_is_sorted_and_ignores_hidden_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            create_scene(root, "b")
            create_scene(root, "a")
            (root / ".cache").mkdir()
            scenes = discover_scenes(root, 2)
            self.assertEqual([name for _, name in scenes], ["a", "b"])
            with self.assertRaises(ValueError):
                discover_scenes(root, 140)

    def test_missing_scene_and_image_raise_instead_of_silent_skip(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "broken").mkdir()
            with self.assertRaises(FileNotFoundError):
                discover_scenes(root)
            directory = create_scene(root, "broken")
            (directory / "images_8/frame_00000.png").unlink()
            with self.assertRaises(FileNotFoundError):
                discover_scenes(root)

    def test_empty_root_is_not_a_successful_benchmark(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(ValueError):
                discover_scenes(name)

    def test_image_path_does_not_double_suffix_or_rewrite_scene_name(self):
        expected = Path("my_images_scene/nerfstudio/images_8/a.png")
        for folder in ("images", "images_2", "images_4", "images_8"):
            self.assertEqual(image_8_path(f"my_images_scene/nerfstudio/{folder}/a.png"), expected)

    def test_distributed_padding_is_removed_before_averaging(self):
        rows, summary = summarize_scenes([[row("a", 10), row("c", 30)],
                                         [row("b", 20), row("a", 10)]], 3)
        self.assertEqual(summary["metrics"]["psnr_ours"], 20)
        self.assertEqual(summary["duplicate_rows_removed"], 1)
        self.assertEqual([r["scene"] for r in rows], ["a", "b", "c"])

    def test_incomplete_or_nonfinite_results_fail(self):
        with self.assertRaises(ValueError):
            summarize_scenes([[row("a", 10)]], 2)
        with self.assertRaises(ValueError):
            summarize_scenes([[row("a", float("nan"))]], 1)
        mismatch = row("a", 10)
        mismatch["target_indices"] = [2]
        with self.assertRaises(ValueError):
            summarize_scenes([[row("a", 10)], [mismatch]], 1)

    def test_result_files_include_metrics_and_protocol(self):
        rows, summary = summarize_scenes([[row("a", 19.3)]], 1)
        with tempfile.TemporaryDirectory() as name:
            write_results(name, rows, summary)
            with (Path(name) / "metrics.json").open() as stream:
                self.assertEqual(json.load(stream)["scene_count"], 1)
            with (Path(name) / "per_scene.csv").open(newline="") as stream:
                output = list(csv.DictReader(stream))
            self.assertEqual(json.loads(output[0]["source_indices"]), [0, 2])
            self.assertEqual(float(output[0]["gir_map_gaussians"]), 100)


class TestConfigHandoff(unittest.TestCase):
    def setUp(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("OmegaConf is not installed")
        self.omega = OmegaConf
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.checkpoint = self.train / "checkpoints/final.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()
        self.config = self.train / ".hydra/config.yaml"
        self.config.parent.mkdir(parents=True)
        self.omega.save(self.omega.create({
            "mode": "train", "checkpointing": {"load": None},
            "dataset": {"dl3dv": {"mode": "train", "roots": ["training_data"],
                                  "ctx_list": [0, 2, 1], "tgt_list": [1]}},
            "model": {"encoder": {"mode": "${mode}", "gir_enabled": True,
                                  "gir_add_gate_enabled": False}},
            "trainer": {"max_steps": 20}, "data_loader": {"test": {"batch_size": 1}},
            "test": {"gir_add_gate_prune_threshold": 0.0},
        }), self.config)
        self.manifest = {"checkpoint": str(self.checkpoint), "global_step": 20,
                         "config": str(self.config)}
        write_json(self.train / "final_checkpoint.json", self.manifest)
        bench = self.root / "dl3dv-140"
        create_scene(bench, "a")
        self.args = SimpleNamespace(train_dir=self.train, bench_root=bench,
                                    expected_scenes=1, devices=4, output_dir=self.root / "results",
                                    config_out=self.root / "test_config/config.yaml", save_compare=False)

    def test_test_reuses_gir_config_and_loads_current_final_checkpoint(self):
        prepare_test(self.args)
        cfg = self.omega.load(self.args.config_out)
        self.assertEqual(cfg.checkpointing.load, str(self.checkpoint))
        self.assertEqual(cfg.model.encoder.mode, "test")
        self.assertFalse(cfg.model.encoder.gir_add_gate_enabled)
        self.assertTrue(cfg.dataset.dl3dv.test_scan_scenes)
        self.assertEqual(cfg.trainer.devices, 4)
        self.assertIsNone(cfg.checkpointing.train_pretrained_weights)

    def test_unfinished_training_does_not_prepare_test(self):
        self.manifest["global_step"] = 19
        write_json(self.train / "final_checkpoint.json", self.manifest)
        with self.assertRaises(ValueError):
            prepare_test(self.args)

    def test_checkpoint_from_another_run_is_rejected(self):
        self.manifest["checkpoint"] = str(self.root / "some_other_run/final.ckpt")
        write_json(self.train / "final_checkpoint.json", self.manifest)
        with self.assertRaises(ValueError):
            prepare_test(self.args)

    def test_invalid_target_split_is_rejected(self):
        cfg = self.omega.load(self.config)
        cfg.dataset.dl3dv.ctx_list = [0, 1, 2]
        self.omega.save(cfg, self.config)
        with self.assertRaises(ValueError):
            prepare_test(self.args)


if __name__ == "__main__":
    unittest.main()
