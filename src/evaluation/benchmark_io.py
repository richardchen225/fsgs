"""Scene discovery and result files shared by the benchmark and its launcher."""

import csv
import json
import math
import os
from pathlib import Path


def image_8_path(path):
    path = Path(path)
    parts = list(path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] in {"images", "images_2", "images_4", "images_8"}:
            parts[index] = "images_8"
            return Path(*parts)
    raise ValueError(f"Frame path has no recognized images directory: {path}")


def discover_scenes(root, expected_count=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")
    scenes = []
    for scene in sorted(root.iterdir()):
        if not scene.is_dir() or scene.name.startswith("."):
            continue
        candidates = [scene / "nerfstudio", scene]
        scene_dir = next((p for p in candidates if (p / "transforms.json").is_file()), None)
        if scene_dir is None:
            raise FileNotFoundError(f"Missing scene/nerfstudio/transforms.json: {scene}")
        with (scene_dir / "transforms.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"No frames in {scene_dir / 'transforms.json'}")
        for frame in frames:
            image = image_8_path(scene_dir / frame["file_path"])
            if not image.is_file():
                raise FileNotFoundError(f"Missing benchmark image: {image}")
        scenes.append((str(scene_dir), scene.name))
    if not scenes:
        raise ValueError(f"No benchmark scenes found under {root}")
    if expected_count is not None and len(scenes) != expected_count:
        raise ValueError(f"Expected {expected_count} scenes, found {len(scenes)} under {root}")
    return scenes


def summarize_scenes(rank_rows, expected_count):
    # DistributedSampler may pad the tail to give all ranks equal work.
    unique = {}
    evaluated_count = 0
    metric_names = ("psnr_ours", "ssim_ours", "lpips_ours")
    for rows in rank_rows:
        for row in rows:
            evaluated_count += 1
            for name in metric_names:
                if not math.isfinite(row[name]):
                    raise ValueError(f"Non-finite {name} for {row['scene']}")
            if row["scene"] in unique:
                previous = unique[row["scene"]]
                for key in ("source_indices", "target_indices", "image_shape"):
                    if previous[key] != row[key]:
                        raise ValueError(f"Inconsistent duplicate scene: {row['scene']}")
            else:
                unique[row["scene"]] = row
    if len(unique) != expected_count or not unique:
        raise ValueError(f"Expected {expected_count} unique results, got {len(unique)}")
    rows = [unique[key] for key in sorted(unique)]
    metrics = {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}
    diagnostic_keys = set.intersection(*(set(row.get("gir", {})) for row in rows))
    diagnostics = {
        key: sum(row["gir"][key] for row in rows) / len(rows)
        for key in sorted(diagnostic_keys)
        if all(math.isfinite(row["gir"][key]) for row in rows)
    }
    summary = {
        "scene_count": len(rows),
        "evaluated_rows": evaluated_count,
        "duplicate_rows_removed": evaluated_count - len(rows),
        "aggregation": "mean of per-scene target-view means",
        "metrics": metrics,
        "gir": diagnostics,
    }
    return rows, summary


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_results(directory, rows, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "source_indices", "target_indices", "image_shape",
              "psnr_ours", "ssim_ours", "lpips_ours"]
    diagnostic_keys = sorted({key for row in rows for key in row.get("gir", {})})
    fields += [f"gir_{key}" for key in diagnostic_keys]
    csv_path = directory / "per_scene.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = {key: row[key] for key in fields if key in row}
            for key in ("source_indices", "target_indices", "image_shape"):
                result[key] = json.dumps(result[key])
            result.update({f"gir_{key}": value for key, value in row.get("gir", {}).items()
                           if math.isfinite(value)})
            writer.writerow(result)
    os.replace(temporary, csv_path)
    # The summary is published last, after the per-scene file is complete.
    write_json(directory / "metrics.json", summary)
