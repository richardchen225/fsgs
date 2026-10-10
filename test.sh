#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash test.sh [CKPT] [OUTPUT_PATH] [EXPERIMENT] [WANDB_NAME] \
#     [DL3DV140_ROOT] [DL3DV140_INDEX]
#
# DL3DV140_ROOT may use the dl3dv_1 layout:
#   <root>/<index item>/<middle directory>/nerfstudio/{transforms.json,images_8/}

CKPT="${1:-/openbayes/home/epoch_15-step_19000.ckpt}"
OUTPUT_PATH="${2:-outputs++/test_dl3dv140}"
EXPERIMENT="${3:-dl3dv}"
WANDB_NAME="${4:-test_dl3dv140}"
DL3DV140_ROOT="${5:-${DL3DV140_ROOT:-/openbayes/input/input0/DL3DV-10K-Benchmark}}"
DL3DV140_INDEX="${6:-${DL3DV140_INDEX:-$DL3DV140_ROOT/dl3dv_140.json}}"

export GPU_NUM="${GPU_NUM:-2}"
export NUM_NODES="${NUM_NODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12345}"
EXPECTED_SCENES="${EXPECTED_SCENES:-140}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "src/main.py" ]]; then
  ENTRYPOINT=(-m src.main)
elif [[ -f "main.py" ]]; then
  ENTRYPOINT=(main.py)
else
  echo "Cannot find main.py or src/main.py under $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi
if [[ ! -d "$DL3DV140_ROOT" ]]; then
  echo "DL3DV-140 root not found: $DL3DV140_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DL3DV140_INDEX" ]]; then
  if [[ -f "$DL3DV140_ROOT/test_index.json" ]]; then
    DL3DV140_INDEX="$DL3DV140_ROOT/test_index.json"
  else
    echo "DL3DV-140 index not found: $DL3DV140_INDEX" >&2
    exit 1
  fi
fi

# DatasetDL3DV expects test_index.json under its configured root. Build a
# temporary indexed view so read-only benchmark data does not need modification.
INDEX_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fsgs-dl3dv140.XXXXXX")"
cleanup() {
  rm -rf -- "$INDEX_ROOT"
}
trap cleanup EXIT

python - "$DL3DV140_ROOT" "$DL3DV140_INDEX" "$INDEX_ROOT" "$EXPECTED_SCENES" <<'PY'
import json
import os
from pathlib import Path
import sys

data_root = Path(sys.argv[1]).resolve()
index_path = Path(sys.argv[2]).resolve()
index_root = Path(sys.argv[3]).resolve()
expected = int(sys.argv[4])

with index_path.open(encoding="utf-8") as stream:
    items = json.load(stream)
if not isinstance(items, list):
    raise TypeError(f"{index_path} must contain a JSON list")
if len(items) != expected:
    raise ValueError(f"Expected {expected} DL3DV-140 entries, found {len(items)}")
if len({str(item) for item in items}) != expected:
    raise ValueError("DL3DV-140 index contains duplicate entries")

staged_items = []
for number, item in enumerate(items):
    item = str(item)
    source = Path(item)
    if not source.is_absolute():
        source = data_root / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {source}")

    # Check both dl3dv_1 and the original benchmark layouts.
    candidates = [path / "nerfstudio" for path in sorted(source.iterdir()) if path.is_dir()]
    candidates.extend((source / "nerfstudio", source))
    scene_dir = next(
        (
            path
            for path in candidates
            if (path / "transforms.json").is_file()
            and (path / "images_8").is_dir()
        ),
        None,
    )
    if scene_dir is None:
        raise FileNotFoundError(
            "Expected <item>/<middle>/nerfstudio or <item>/nerfstudio with "
            f"transforms.json and images_8: {source}"
        )

    with (scene_dir / "transforms.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {scene_dir / 'transforms.json'}")
    for frame in frames:
        frame_path = str(scene_dir / frame["file_path"]).replace(
            "images", "images_8"
        )
        if not Path(frame_path).is_file():
            raise FileNotFoundError(f"Missing images_8 frame: {frame_path}")

    # Preserve short, stable scene identifiers while pointing at the real data.
    relative = Path("scenes") / f"{number:03d}_{source.name}"
    destination = index_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, destination, target_is_directory=True)
    staged_items.append(relative.as_posix())

with (index_root / "test_index.json").open("w", encoding="utf-8") as stream:
    json.dump(staged_items, stream, indent=2)
    stream.write("\n")

print(f"DL3DV-140 preflight passed: {len(staged_items)} scenes")
print(f"Temporary dataset index: {index_root / 'test_index.json'}")
PY

mkdir -p -- "$OUTPUT_PATH"

torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$GPU_NUM" \
  --rdzv_id=test \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  "${ENTRYPOINT[@]}" \
  +experiment="$EXPERIMENT" \
  +hydra.job.config.store_config=false \
  mode=test \
  model.encoder.mode=test \
  wandb.mode=offline \
  wandb.name="$WANDB_NAME" \
  trainer.devices="$GPU_NUM" \
  trainer.num_nodes="$NUM_NODES" \
  data_loader.test.batch_size=1 \
  dataset.dl3dv.name=dl3dv_1 \
  dataset.dl3dv.mode=test \
  "dataset.dl3dv.roots=[\"$INDEX_ROOT\"]" \
  dataset.dl3dv.test_scan_scenes=false \
  dataset.dl3dv.test_expected_scenes="$EXPECTED_SCENES" \
  test.compute_scores=true \
  test.save_metrics=true \
  test.output_path="$OUTPUT_PATH" \
  checkpointing.load="$CKPT"
