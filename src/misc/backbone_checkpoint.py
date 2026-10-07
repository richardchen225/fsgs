"""Prevent silent use of GS checkpoints trained with a different backbone."""


def backbone_signature(encoder_cfg):
    name = getattr(encoder_cfg, "reconstruction_backbone", "zipmap")
    return {"name": name}


def validate_backbone_checkpoint(checkpoint, encoder_cfg, *, allow_transfer=False):
    expected = backbone_signature(encoder_cfg)
    actual = checkpoint.get("reconstruction_backbone")
    if actual is None:
        state = checkpoint.get("state_dict", checkpoint)
        keys = [key.removeprefix("model.") for key in state]
        if any(key.startswith("encoder.aggregator.aggregator.") for key in keys):
            actual = {"name": "zipmap"}
    if actual is None:
        return
    mismatch = actual.get("name") != expected["name"]
    if mismatch:
        message = f"Checkpoint backbone {actual} does not match configured {expected}."
        if not allow_transfer:
            raise RuntimeError(message)
        print(f"{message} Transferring head initialization only; heads must be retrained.")
