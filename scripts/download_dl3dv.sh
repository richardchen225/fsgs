#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/download_dl3dv.sh <output_dir>

Examples:
  bash scripts/download_dl3dv.sh datasets/dl3dv
  bash scripts/download_dl3dv.sh /data/DL3DV-ALL-480P

This script downloads DL3DV/DL3DV-ALL-480P images+poses from 1K through 11K.
You must pass the output directory explicitly.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "Missing output directory." >&2
  usage
  exit 1
fi

OUTPUT_DIR="$1"
RESOLUTION="480P"
FILE_TYPE="images+poses"
SUBSETS=(1K 2K 3K 4K 5K 6K 7K 8K 9K 10K 11K)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_INDEX_SOURCE="${DL3DV_TEST_INDEX_PATH:-$REPO_ROOT/test_index.json}"

mkdir -p "$OUTPUT_DIR"

python -m pip install 'huggingface_hub==0.23.4' pandas tqdm

DOWNLOAD_SCRIPT="$OUTPUT_DIR/dl3dv_download.py"
RAW_URL="https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py"

if command -v wget >/dev/null 2>&1; then
  wget -O "$DOWNLOAD_SCRIPT" "$RAW_URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L "$RAW_URL" -o "$DOWNLOAD_SCRIPT"
else
  echo "wget or curl is required to fetch the official DL3DV download script." >&2
  exit 1
fi

for subset in "${SUBSETS[@]}"; do
  echo "Downloading DL3DV/DL3DV-ALL-480P subset $subset to $OUTPUT_DIR"
  python "$DOWNLOAD_SCRIPT" \
    --odir "$OUTPUT_DIR" \
    --subset "$subset" \
    --resolution "$RESOLUTION" \
    --file_type "$FILE_TYPE" \
    --clean_cache
done

python - "$OUTPUT_DIR" "$TEST_INDEX_SOURCE" <<'PY'
import json
import sys
from pathlib import Path
import shutil

root = Path(sys.argv[1]).resolve()
test_index_source = Path(sys.argv[2]).resolve()
subsets = {f"{idx}K" for idx in range(1, 12)}
test_items = set()
if test_index_source.exists():
    with test_index_source.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise TypeError(f"{test_index_source} must be a JSON array")
    test_items = {str(item) for item in loaded}

items = set()
for transforms in sorted(root.rglob("transforms.json")):
    scene_dir = transforms.parent
    if not (scene_dir / "images_8").is_dir():
        continue
    relative_parts = scene_dir.relative_to(root).parts
    if len(relative_parts) >= 2 and relative_parts[0] in subsets:
        items.add(Path(*relative_parts[:2]).as_posix())
    else:
        items.add(scene_dir.relative_to(root).as_posix())

index_path = root / "train_index.json"
test_index_path = root / "test_index.json"
if items:
    items = sorted(item for item in items if item not in test_items)
    index_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {index_path} with {len(items)} scenes.")
else:
    print(
        "No DL3DV scenes were found for train_index.json. "
        "Check that downloaded images+poses contain <subset>/<scene_name>/<scene_data_dir>/images_8.",
        file=sys.stderr,
    )

if test_index_source.exists():
    shutil.copy2(test_index_source, test_index_path)
    print(f"Copied {test_index_source} to {test_index_path}.")
PY

echo "DL3DV/DL3DV-ALL-480P 1K-11K download finished under $OUTPUT_DIR"
