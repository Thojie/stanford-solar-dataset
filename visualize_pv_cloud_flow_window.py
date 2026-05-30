from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


def parse_dt(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S")


def load_times(root: Path, split: str) -> np.ndarray:
    return np.load(root / "output_folder_v3" / f"times_{split}.npy", allow_pickle=True)


def find_window_indices(times: np.ndarray, start: datetime, end: datetime) -> list[int]:
    return [idx for idx, ts in enumerate(times) if start <= ts <= end]


def choose_split(root: Path, start: datetime, end: datetime, preferred: str) -> tuple[str, np.ndarray, list[int]]:
    splits = [preferred] if preferred != "auto" else ["test", "trainval"]
    for split in splits:
        times = load_times(root, split)
        indices = find_window_indices(times, start, end)
        if indices:
            return split, times, indices
    raise ValueError(f"No samples found for window {start} -> {end} in split={preferred!r}")


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    return image


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    u = flow[0].astype(np.float32)
    v = flow[1].astype(np.float32)
    magnitude = np.sqrt(u * u + v * v)
    angle = np.arctan2(v, u)

    hue = (angle + np.pi) / (2.0 * np.pi)
    saturation = np.clip(magnitude / (np.percentile(magnitude, 99) + 1e-6), 0.0, 1.0)
    value = np.ones_like(saturation)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv_to_rgb(hsv)


def build_figure(times: np.ndarray, samples: list[dict], output_path: Path, title: str) -> None:
    n = len(samples)
    fig = plt.figure(figsize=(18, max(4.5, 2.8 + 2.6 * n)), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(nrows=n + 1, ncols=4, height_ratios=[1.15] + [1] * n, hspace=0.28, wspace=0.06)

    ax_curve = fig.add_subplot(grid[0, :])
    pv_values = [sample["pv"] for sample in samples]
    x = np.arange(n)
    ax_curve.plot(x, pv_values, color="#2a6fdb", linewidth=2.0, marker="o", markersize=4)
    ax_curve.set_title(title)
    ax_curve.set_ylabel("PV")
    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels([sample["timestamp"].strftime("%H:%M:%S") for sample in samples], rotation=0)
    ax_curve.grid(alpha=0.25)

    for pos, sample in enumerate(samples):
        ax_curve.scatter([pos], [sample["pv"]], color="#d62728", s=28, zorder=3)

    for row, sample in enumerate(samples, start=1):
        ax_text = fig.add_subplot(grid[row, 0])
        ax_global = fig.add_subplot(grid[row, 1])
        ax_local = fig.add_subplot(grid[row, 2])
        ax_flow = fig.add_subplot(grid[row, 3])

        ax_text.axis("off")
        ax_text.text(
            0.02,
            0.65,
            sample["timestamp"].strftime("%Y-%m-%d\n%H:%M:%S"),
            fontsize=11,
            fontweight="bold",
            va="center",
        )
        ax_text.text(0.02, 0.22, f"PV: {sample['pv']:.3f}", fontsize=10, va="center")

        ax_global.imshow(sample["global_image"])
        ax_global.set_title("Global")
        ax_global.set_xticks([])
        ax_global.set_yticks([])

        ax_local.imshow(sample["local_image"])
        ax_local.set_title("Local")
        ax_local.set_xticks([])
        ax_local.set_yticks([])

        ax_flow.imshow(sample["flow_rgb"])
        ax_flow.set_title("Flow")
        ax_flow.set_xticks([])
        ax_flow.set_yticks([])

    fig.savefig(output_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synchronized PV, global cloud, local cloud and optical flow.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Project root directory.")
    parser.add_argument("--date", default="2019-10-06", help="Target date, YYYY-MM-DD.")
    parser.add_argument("--start", default="12:25:00", help="Window start time, HH:MM:SS.")
    parser.add_argument("--end", default="12:35:00", help="Window end time, HH:MM:SS.")
    parser.add_argument("--split", default="auto", choices=["auto", "trainval", "test"], help="Which split to search.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving.")
    args = parser.parse_args()

    root = args.root
    start = parse_dt(args.date, args.start)
    end = parse_dt(args.date, args.end)
    if end < start:
        raise ValueError("--end must not be earlier than --start")

    split, times, indices = choose_split(root, start, end, args.split)

    dual_path = root / "output_folder_v3" / "2019_dataset_dual_V3.h5"
    flow_path = root / "output_folder_v3" / "2019_dataset_flow_V3.h5"

    samples: list[dict] = []
    with h5py.File(dual_path, "r") as dual_h5, h5py.File(flow_path, "r") as flow_h5:
        dual_group = dual_h5["trainval"]
        flow_group = flow_h5["trainval"]

        for idx in indices:
            global_image = chw_to_hwc(dual_group["global_images_log"][idx])
            local_image = chw_to_hwc(dual_group["local_images_log"][idx])
            flow_rgb = flow_to_rgb(flow_group["global_flow_log"][idx])

            samples.append(
                {
                    "timestamp": times[idx],
                    "pv": float(dual_group["pv_log"][idx]),
                    "global_image": global_image,
                    "local_image": local_image,
                    "flow_rgb": flow_rgb,
                }
            )

    output_path = args.output
    if output_path is None:
        safe_start = start.strftime("%Y%m%d_%H%M%S")
        safe_end = end.strftime("%H%M%S")
        output_path = root / "output_folder_v3" / f"viz_{safe_start}_{safe_end}_{split}.png"

    title = f"PV / Global Cloud / Local Cloud / Flow | {split} | {start.strftime('%Y-%m-%d %H:%M:%S')} -> {end.strftime('%H:%M:%S')}"
    build_figure(times, samples, output_path, title)

    print(f"split = {split}")
    print(f"matched {len(indices)} samples")
    for idx, sample in zip(indices, samples):
        print(f"{idx}: {sample['timestamp']} | PV={sample['pv']:.3f}")
    print(f"saved to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()