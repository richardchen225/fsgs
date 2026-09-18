"""Prevent silent use of GS checkpoints trained with a different backbone."""


def backbone_signature(encoder_cfg):
    name = getattr(encoder_cfg, "reconstruction_backbone", "zipmap")
    signature = {"name": name}
    if name == "abot":
        signature["feature_layers"] = list(encoder_cfg.abot_feature_layers)
        signature["pose_mode"] = getattr(encoder_cfg, "abot_pose_mode", "two_pass")
    return signature


def validate_backbone_checkpoint(checkpoint, encoder_cfg, *, allow_transfer=False):
    expected = backbone_signature(encoder_cfg)
    actual = checkpoint.get("reconstruction_backbone")
    if actual is None:
        state = checkpoint.get("state_dict", checkpoint)
        keys = [key.removeprefix("model.") for key in state]
        if any(key.startswith("encoder.aggregator.network.") for key in keys):
            actual = {"name": "abot"}
        elif any(key.startswith("encoder.aggregator.aggregator.") for key in keys):
            actual = {"name": "zipmap"}
    if actual is None:
        if expected["name"] == "abot" and not allow_transfer:
            raise RuntimeError(
                "This checkpoint does not identify its reconstruction backbone. "
                "ABot test/resume requires an ABot training checkpoint. "
                "Use WM/legacy head weights only to initialize trainable heads."
            )
        return
    mismatch = actual.get("name") != expected["name"]
    if "feature_layers" in actual:
        mismatch |= actual["feature_layers"] != expected.get("feature_layers")
    if mismatch:
        message = f"Checkpoint backbone {actual} does not match configured {expected}."
        if not allow_transfer:
            raise RuntimeError(message)
        print(f"{message} Transferring head initialization only; heads must be retrained.")
    elif expected["name"] == "abot" and actual.get("pose_mode", "single_pass") != expected["pose_mode"]:
        # The head shapes are compatible; allow deliberate protocol comparisons,
        # but do not imply that this continues the same training experiment.
        print(
            "Warning: ABot camera protocol changed from "
            f"{actual.get('pose_mode', 'single_pass')} to {expected['pose_mode']}. "
            "Target poses/losses can change. Use single_pass to reproduce an old run."
        )
