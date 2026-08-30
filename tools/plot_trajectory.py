#!/usr/bin/env python3
"""Plot pipeline pose CSV trajectories with lines between adjacent samples.

This is a standalone inspection utility and is not part of stages 01--08.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath


CAMERA_MARKER = MplPath(
    [
        (1.0, 0.0),
        (0.05, 0.46),
        (-0.72, 0.28),
        (-0.38, 0.0),
        (-0.72, -0.28),
        (0.05, -0.46),
        (1.0, 0.0),
    ],
    [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ],
)

def oriented_marker(angle_rad: float) -> MarkerStyle:
    marker = MarkerStyle(CAMERA_MARKER)
    marker._transform.rotate(angle_rad)
    return marker


def infer_chunk_index(path: Path) -> int | None:
    match = re.search(r"chunk_(\d+)_poses\.csv$", path.name)
    return int(match.group(1)) if match else None


def overlap_mask(scene_frames: list[float], csv_path: Path, chunk_frames: int | None) -> list[bool]:
    chunk_index = infer_chunk_index(csv_path)
    if chunk_index is None or chunk_frames is None or chunk_frames <= 0:
        return [False for _ in scene_frames]
    logical_start = chunk_index * chunk_frames
    logical_end = logical_start + chunk_frames
    return [frame < logical_start or frame >= logical_end for frame in scene_frames]


def read_pose_csv(path: Path) -> dict[str, list[float]]:
    cols = {"scene_frame": [], "source_timestamp": [], "x": [], "y": [], "z": [], "qx": [], "qy": [], "qz": [], "qw": []}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = set(cols)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            cols["scene_frame"].append(float(row["scene_frame"]))
            cols["source_timestamp"].append(float(row["source_timestamp"]))
            cols["x"].append(float(row["x"]))
            cols["y"].append(float(row["y"]))
            cols["z"].append(float(row["z"]))
            cols["qx"].append(float(row["qx"]))
            cols["qy"].append(float(row["qy"]))
            cols["qz"].append(float(row["qz"]))
            cols["qw"].append(float(row["qw"]))
    if not cols["source_timestamp"]:
        raise ValueError(f"{path} has no pose rows")
    return cols


def equalize_xy(ax, xs: list[float], ys: list[float]) -> None:
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = span * 0.02
    half = span / 2 + pad
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def camera_forward_world(qx: float, qy: float, qz: float, qw: float) -> dict[str, float]:
    # VGGT-Omega writes camera-to-world rotations converted from OpenCV
    # world-to-camera extrinsics. In OpenCV camera coordinates, +Z is the
    # optical/view direction, so R_c2w @ [0, 0, 1] is the camera look vector
    # in the same world coordinates as x/y/z positions.
    r = quaternion_to_matrix(qx, qy, qz, qw)
    return {"x": r[0][2], "y": r[1][2], "z": r[2][2]}


def projected_yaw_pitch(
    forward: dict[str, float],
    plane_x: str,
    plane_y: str,
    height_axis: str,
    height_sign: float = 1.0,
) -> tuple[float, float]:
    # Use the exact axes plotted below. For OpenCV/VGGT-style camera
    # coordinates, x-z is the natural walking plane and -y is up.
    fx = forward[plane_x]
    fy = forward[plane_y]
    fh = height_sign * forward[height_axis]
    yaw = math.atan2(fy, fx)
    pitch = math.atan2(fh, max(math.hypot(fx, fy), 1e-12))
    return yaw, pitch


def quaternion_plot_yaw_pitch(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float]:
    return projected_yaw_pitch(camera_forward_world(qx, qy, qz, qw), "x", "z", "y", -1.0)

def plot_pose_csv(
    csv_path: Path,
    out_path: Path,
    show_orientation: bool,
    marker_step: int,
    marker_size: float,
    chunk_frames: int | None,
) -> None:
    data = read_pose_csv(csv_path)
    t = data["source_timestamp"]
    x = data["x"]
    y = data["y"]
    z = data["z"]
    plane_x = x
    plane_y = z
    height = [-value for value in y]
    scene_frames = data["scene_frame"]
    is_overlap = overlap_mask(scene_frames, csv_path, chunk_frames)
    valid_indices = [i for i, value in enumerate(is_overlap) if not value]
    start_idx = valid_indices[0] if valid_indices else 0
    end_idx = valid_indices[-1] if valid_indices else len(x) - 1
    yaw_pitch = [
        quaternion_plot_yaw_pitch(qx, qy, qz, qw)
        for qx, qy, qz, qw in zip(data["qx"], data["qy"], data["qz"], data["qw"])
    ]
    yaw = [item[0] for item in yaw_pitch]
    pitch = [item[1] for item in yaw_pitch]

    fig, (ax_xy, ax_z) = plt.subplots(
        2,
        1,
        figsize=(14, 16),
        dpi=180,
        gridspec_kw={"height_ratios": [5, 1]},
        constrained_layout=True,
    )
    cmap = "viridis"

    ax_xy.plot(plane_x, plane_y, color="0.55", linewidth=1.1, alpha=0.45, zorder=1)
    norm = plt.Normalize(min(t), max(t))
    colors = plt.get_cmap(cmap)(norm(t))
    if show_orientation:
        step = max(1, marker_step)
        idx = list(range(0, len(x), step))
        if idx[-1] != len(x) - 1:
            idx.append(len(x) - 1)
        for i in idx:
            ax_xy.scatter(
                [plane_x[i]],
                [plane_y[i]],
                c=["#d95f02" if is_overlap[i] else colors[i]],
                s=marker_size,
                marker=oriented_marker(yaw[i]),
                linewidths=0,
                alpha=0.95,
                zorder=3,
            )
    ax_xy.scatter([plane_x[start_idx]], [plane_y[start_idx]], c="black", s=96, marker="o", label="valid start", zorder=4)
    ax_xy.scatter([plane_x[end_idx]], [plane_y[end_idx]], c="red", s=110, marker="X", label="valid end", zorder=4)
    if any(is_overlap):
        ax_xy.scatter([], [], c="#d95f02", s=marker_size, marker=oriented_marker(0), label="overlap")
    ax_xy.set_title(csv_path.stem)
    ax_xy.set_xlabel("x (VGGT/OpenCV world)")
    ax_xy.set_ylabel("z forward/depth (VGGT/OpenCV world)")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best", frameon=False)
    equalize_xy(ax_xy, plane_x, plane_y)

    ax_z.plot(t, height, color="0.55", linewidth=1.1, alpha=0.45, zorder=1)
    if show_orientation:
        step = max(1, marker_step)
        idx = list(range(0, len(t), step))
        if idx[-1] != len(t) - 1:
            idx.append(len(t) - 1)
        for i in idx:
            ax_z.scatter(
                [t[i]],
                [height[i]],
                c=["#d95f02" if is_overlap[i] else colors[i]],
                s=marker_size,
                marker=oriented_marker(pitch[i]),
                linewidths=0,
                alpha=0.95,
                zorder=3,
            )
    ax_z.scatter([t[start_idx]], [height[start_idx]], c="black", s=42, marker="o", zorder=4)
    ax_z.scatter([t[end_idx]], [height[end_idx]], c="red", s=52, marker="X", zorder=4)
    ax_z.annotate("start", (t[start_idx], height[start_idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax_z.annotate("end", (t[end_idx], height[end_idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax_z.set_xlabel("time (s)")
    ax_z.set_ylabel("-y up (VGGT/OpenCV world)")
    ax_z.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_xy, ax_z], shrink=0.72, fraction=0.035, pad=0.02)
    cbar.set_label("time (s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def collect_csvs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            direct = item / "poses.csv"
            if direct.exists():
                paths.append(direct)
            else:
                paths.extend(sorted(item.rglob("*_poses.csv")))
                paths.extend(sorted(item.rglob("poses.csv")))
        else:
            paths.append(item)
    return sorted(dict.fromkeys(paths))


def output_path(csv_path: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return csv_path.with_suffix(".png")
    return out_dir / f"{csv_path.stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="pose CSV files or directories containing *_poses.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="optional directory for generated PNG files")
    parser.add_argument("--no-orientation", action="store_true", help="hide yaw/pitch orientation markers")
    parser.add_argument("--marker-step", "--arrow-step", type=int, default=1, help="draw one orientation marker every N poses")
    parser.add_argument("--marker-size", "--arrow-scale", type=float, default=288.0, help="orientation camera marker area in points^2")
    parser.add_argument("--chunk-frames", type=int, default=60, help="logical frames per chunk for overlap coloring")
    args = parser.parse_args()

    csvs = collect_csvs(args.inputs)
    if not csvs:
        raise SystemExit("no pose CSV files found")
    for csv_path in csvs:
        out_path = output_path(csv_path, args.out_dir)
        plot_pose_csv(csv_path, out_path, not args.no_orientation, args.marker_step, args.marker_size, args.chunk_frames)
        print(out_path)


if __name__ == "__main__":
    main()
