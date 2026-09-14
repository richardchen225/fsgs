"""Check the checkpoint hooks without importing CUDA-only model dependencies."""

import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def load_hooks():
    source = Path(__file__).resolve().parents[1] / "src/model/model_wrapper.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wrapper = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelWrapper"
    )
    methods = [
        node for node in wrapper.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"on_save_checkpoint", "on_load_checkpoint"}
    ]
    namespace = {"Any": object}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["on_save_checkpoint"], namespace["on_load_checkpoint"]


SAVE, LOAD = load_hooks()


class CheckpointLossStateTests(unittest.TestCase):
    def test_save_preserves_model_and_training_state(self):
        model_weight, loss_weight = object(), object()
        state = OrderedDict([
            ("model.encoder.weight", model_weight),
            ("losses.3.depth_anything.weight", loss_weight),
            ("model.losses.weight", model_weight),
        ])
        state._metadata = {"": {"version": 1}}
        optimizer_states, schedulers = [object()], [object()]
        checkpoint = {
            "state_dict": state,
            "optimizer_states": optimizer_states,
            "lr_schedulers": schedulers,
            "global_step": 123,
        }
        SAVE(None, checkpoint)
        self.assertEqual(list(state), ["model.encoder.weight", "model.losses.weight"])
        self.assertIs(state["model.encoder.weight"], model_weight)
        self.assertEqual(state._metadata, {"": {"version": 1}})
        self.assertIs(checkpoint["optimizer_states"], optimizer_states)
        self.assertIs(checkpoint["lr_schedulers"], schedulers)
        self.assertEqual(checkpoint["global_step"], 123)

    def test_load_uses_initialized_losses_without_filling_missing_model(self):
        loss_weight = object()
        losses = Mock()
        losses.state_dict.return_value = {"losses.3.weight": loss_weight}
        checkpoint = {"state_dict": {"model.present": object()}}
        LOAD(SimpleNamespace(losses=losses), checkpoint)
        losses.state_dict.assert_called_once_with(prefix="losses.")
        self.assertIs(checkpoint["state_dict"]["losses.3.weight"], loss_weight)
        self.assertEqual(set(checkpoint["state_dict"]), {"model.present", "losses.3.weight"})

    def test_legacy_checkpoint_keeps_saved_loss_weights(self):
        saved_weight = object()
        losses = Mock()
        losses.state_dict.return_value = {"losses.3.weight": object()}
        checkpoint = {"state_dict": {"losses.3.weight": saved_weight}}
        LOAD(SimpleNamespace(losses=losses), checkpoint)
        self.assertIs(checkpoint["state_dict"]["losses.3.weight"], saved_weight)

    def test_no_persistent_loss_state_is_supported(self):
        losses = Mock()
        losses.state_dict.return_value = {}
        checkpoint = {"state_dict": {"model.weight": object()}}
        before = checkpoint["state_dict"].copy()
        SAVE(None, checkpoint)
        LOAD(SimpleNamespace(losses=losses), checkpoint)
        self.assertEqual(checkpoint["state_dict"], before)

    def test_strict_pytorch_restore_when_torch_is_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        wrapper = torch.nn.Module()
        wrapper.model = torch.nn.Linear(2, 2)
        wrapper.losses = torch.nn.ModuleList([torch.nn.Linear(2, 1)])
        checkpoint = {"state_dict": wrapper.state_dict()}
        SAVE(wrapper, checkpoint)
        self.assertFalse(any(k.startswith("losses.") for k in checkpoint["state_dict"]))
        LOAD(wrapper, checkpoint)
        wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint["state_dict"]["model.weight"]
        LOAD(wrapper, checkpoint)
        with self.assertRaises(RuntimeError):
            wrapper.load_state_dict(checkpoint["state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
