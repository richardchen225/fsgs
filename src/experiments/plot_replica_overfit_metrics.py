from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_evaluations(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Resuming records the resume step once more in the new run. Keep the last
    # value for each step so the curve remains continuous without duplicates.
    rows_by_step = {int(row["step"]): row for row in rows}
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _read_training(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def plot_metrics(output_dir: str | Path, output_path: str | Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    evaluations = _read_evaluations(output_dir / "eval_metrics.jsonl")
    training = _read_training(output_dir / "metrics.csv")
    if not evaluations and not training:
        raise FileNotFoundError(
            f"No eval_metrics.jsonl or metrics.csv found under {output_dir}."
        )

    figure, axes = plt.subplots(2, 3, figsize=(19, 9), constrained_layout=True)
    psnr_ax, loss_ax, mask_ax, update_ax, depth_ax, spare_ax = axes.ravel()

    if evaluations:
        steps = [row["step"] for row in evaluations]
        psnr_series = (
            ("base_first_psnr", "Base first input"),
            ("global_first_psnr", "Final map first input"),
            ("global_current_psnr", "Final map last input"),
            ("global_current_overlap_psnr", "Final map last overlap"),
            ("global_current_hole_psnr", "Final map new holes"),
            ("heldout_loss_psnr", "Held-out loss view"),
            ("heldout_test_psnr", "Held-out test view"),
            ("heldout_psnr", "Held-out frame (legacy)"),
            ("input_mean_psnr", "Input mean"),
        )
        for key, label in psnr_series:
            points = [(row["step"], row[key]) for row in evaluations if key in row]
            if points:
                psnr_ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=label,
                )
        psnr_ax.set_title("Evaluation PSNR")
        psnr_ax.set_xlabel("Step")
        psnr_ax.set_ylabel("PSNR (dB)")
        psnr_ax.grid(alpha=0.25)
        psnr_ax.legend(fontsize=8, ncol=2)

        for key, label in (
            ("old_coverage_ratio", "Old-map coverage"),
            ("depth_consistent_ratio", "Depth consistent"),
            ("current_in_front_ratio", "Current in front"),
            ("old_in_front_ratio", "Old map in front"),
            ("overlap_ratio", "Overlap"),
            ("bad_ratio", "Bad pixels"),
            ("append_ratio", "Appended holes"),
        ):
            points = [(row["step"], row[key]) for row in evaluations if key in row]
            if points:
                mask_ax.plot(
                    [point[0] for point in points],
                    [100.0 * point[1] for point in points],
                    marker="o",
                    markersize=3,
                    label=label,
                )
        mask_ax.set_title("Evidence masks")
        mask_ax.set_xlabel("Step")
        mask_ax.set_ylabel("Pixels (%)")
        mask_ax.grid(alpha=0.25)
        mask_ax.legend(fontsize=8)

        for key, label in (
            ("depth_dav3_gt_abs_rel_aligned", "DAV3 GT AbsRel aligned"),
            ("depth_dav3_gt_rmse_aligned", "DAV3 GT RMSE aligned"),
            ("depth_dav3_gt_log_l1_aligned", "DAV3 GT log-L1 aligned"),
            ("depth_dav3_gt_abs_rel_raw", "DAV3 GT AbsRel raw"),
            ("depth_dav3_gt_scale", "DAV3 scale to GT"),
            ("depth_dav3_structure_log_l1", "Student/DAV3 structure"),
            ("depth_dav3_absolute_log_scale", "Student/DAV3 absolute scale"),
        ):
            points = [(row["step"], row[key]) for row in evaluations if key in row]
            if points:
                depth_ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label=label,
                )
        depth_ax.set_title("DAV3 vs GT depth")
        depth_ax.set_xlabel("Step")
        depth_ax.set_ylabel("Error / fitted scale")
        depth_ax.grid(alpha=0.25)
        depth_ax.legend(fontsize=8)

        update_ax.plot(
            steps,
            [row["unique_updated_gs"] for row in evaluations],
            marker="o",
            markersize=3,
            color="tab:red",
            label="Unique updated GS",
        )

    if training:
        train_steps = [row["step"] for row in training]
        for key, label in (
            ("loss", "Total"),
            ("loss_base_first", "Base first input"),
            ("loss_updated_current", "Updated current full image"),
            ("loss_new_holes", "New GS holes mean"),
            ("loss_replay_history", "Replay history mean"),
            ("loss_depth", "DAV3 depth total"),
            ("loss_depth_structure", "DAV3 structure"),
            ("loss_depth_absolute_log_scale", "DAV3 absolute log-scale"),
            ("loss_camera", "Camera total"),
            ("loss_camera_translation", "Camera translation"),
            ("loss_camera_rotation", "Camera rotation"),
            ("loss_camera_focal", "Camera raw FOV"),
            ("loss_heldout", "Held-out RGB"),
            ("loss_opacity_decay_budget", "Opacity decay budget"),
            ("loss_updated_first", "Replay history mean (legacy)"),
        ):
            points = [(row["step"], row[key]) for row in training if key in row]
            if points:
                loss_ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    linewidth=1.0,
                    label=label,
                )
        loss_ax.set_yscale("log")
        loss_ax.set_title("Training losses")
        loss_ax.set_xlabel("Step")
        loss_ax.set_ylabel("Loss (log scale)")
        loss_ax.grid(alpha=0.25)
        loss_ax.legend(fontsize=8)

        if not evaluations:
            update_ax.plot(
                train_steps,
                [row["unique_updated_gs"] for row in training],
                color="tab:red",
                linewidth=1.0,
                label="Unique updated GS",
            )
        grad_ax = update_ax.twinx()
        grad_points = [
            (row["step"], row["grad_norm"])
            for row in training
            if "grad_norm" in row and row["grad_norm"] > 0
        ]
        if grad_points:
            grad_ax.plot(
                [point[0] for point in grad_points],
                [point[1] for point in grad_points],
                color="tab:gray",
                alpha=0.55,
                linewidth=0.8,
                label="Gradient norm",
            )
            grad_ax.set_yscale("log")
            grad_ax.set_ylabel("Gradient norm (log scale)")

    update_ax.set_title("Update support and gradients")
    update_ax.set_xlabel("Step")
    update_ax.set_ylabel("Unique GS")
    update_ax.grid(alpha=0.25)
    update_ax.legend(fontsize=8, loc="upper left")
    decay_rows = training if training else evaluations
    for key, label in (
        ("opacity_decay_candidate_probability", "Decay probability"),
        ("opacity_decay_mass_ratio", "Opacity mass reduction"),
    ):
        points = [
            (row["step"], row[key])
            for row in decay_rows
            if key in row
        ]
        if points:
            spare_ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=1.2,
                label=label,
            )
    spare_ax.set_title("Top-4 opacity decay")
    spare_ax.set_xlabel("Step")
    spare_ax.set_ylabel("Probability / ratio")
    spare_ax.grid(alpha=0.25)
    if spare_ax.lines:
        spare_ax.legend(fontsize=8)

    destination = Path(output_path) if output_path is not None else output_dir / "metric_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_residual_update_metrics(
    output_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    evaluations = _read_evaluations(output_dir / "eval_metrics.jsonl")
    training = _read_training(output_dir / "metrics.csv")
    rank_probe = "rank1_effective_mean_shift"
    rows = (
        training
        if any(rank_probe in row for row in training)
        else evaluations
    )
    if not rows or not any(rank_probe in row for row in rows):
        return None

    figure, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    axes = axes.ravel()
    rank_indices = [
        rank_idx
        for rank_idx in range(1, 9)
        if any(f"rank{rank_idx}_valid_ratio" in row for row in rows)
    ]
    rank_colors = plt.get_cmap("tab10").colors[: len(rank_indices)]

    def points(key: str, scale: float = 1.0) -> tuple[list[float], list[float]]:
        selected = [
            (float(row["step"]), float(row[key]) * scale)
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        return (
            [point[0] for point in selected],
            [point[1] for point in selected],
        )

    def plot_rank_metric(
        axis,
        suffix: str,
        title: str,
        ylabel: str,
        symlog: bool = True,
    ) -> None:
        plotted = False
        for rank_idx, color in zip(rank_indices, rank_colors):
            x, y = points(f"rank{rank_idx}_{suffix}")
            if not x:
                continue
            axis.plot(x, y, color=color, linewidth=1.2, label=f"Top {rank_idx}")
            plotted = True
        if symlog:
            axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)

    plot_rank_metric(
        axes[0], "effective_mean_shift", "Effective mean shift", "Distance"
    )
    plot_rank_metric(
        axes[1], "effective_rotation", "Effective rotation", "Radians"
    )
    plot_rank_metric(
        axes[2], "effective_log_scale", "Effective log-scale", "Magnitude"
    )
    plot_rank_metric(
        axes[3],
        "effective_opacity_logit",
        "Effective opacity-logit",
        "Magnitude",
    )
    plot_rank_metric(
        axes[4], "effective_harmonics", "Effective SH residual", "RMS"
    )
    plot_rank_metric(
        axes[5],
        "update_multiplier",
        "Gate, damping and rank multiplier",
        "Multiplier",
        symlog=False,
    )

    for rank_idx, color in zip(rank_indices, rank_colors):
        x, y = points(f"rank{rank_idx}_valid_ratio", scale=100.0)
        if x:
            axes[6].plot(
                x, y, color=color, linewidth=1.2, label=f"Top {rank_idx} valid"
            )
    x, y = points("map_updated_ratio", scale=100.0)
    if x:
        axes[6].plot(
            x,
            y,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Updated GS",
        )
    axes[6].set_title("Update support")
    axes[6].set_xlabel("Step")
    axes[6].set_ylabel("Ratio (%)")
    axes[6].grid(alpha=0.25)
    axes[6].legend(fontsize=8)

    map_series = (
        ("mean_shift", "Mean shift"),
        ("rotation", "Rotation"),
        ("scale_relative", "Relative scale"),
        ("opacity", "Opacity"),
        ("harmonics", "SH RMS"),
    )
    for axis, reduction, title in (
        (axes[7], "mean", "Actual old-GS change: mean"),
        (axes[8], "max", "Actual old-GS change: max"),
    ):
        for name, label in map_series:
            x, y = points(f"map_{name}_{reduction}")
            if x:
                axis.plot(x, y, linewidth=1.2, label=label)
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel("Change magnitude")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    destination = (
        Path(output_path)
        if output_path is not None
        else output_dir / "residual_update_curves.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Experiment directory containing metric logs.")
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    args = parser.parse_args()
    path = plot_metrics(args.output_dir, args.output)
    print(f"Saved metric curves to {path}")
    residual_path = plot_residual_update_metrics(args.output_dir)
    if residual_path is not None:
        print(f"Saved residual update curves to {residual_path}")


if __name__ == "__main__":
    main()
