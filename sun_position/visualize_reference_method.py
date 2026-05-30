from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sun_posion_identification import sun_position


CONFIG = {
    "hdf5_path": "dataset/2017_2019_images_pv_processed.hdf5",
    "data_dir": "dataset",
    "split": "trainval",
    "year": 2019,
    "months": list(range(1, 13)),  
    "num_samples": 20,
    "stride": 30,
    "output_dir": "experiments/reference_method_viz_2019",
}


def load_times_and_indices(data_dir: str, split: str, year: int):
    times_path = f"{data_dir}/times_{split}.npy"
    times_all = np.load(times_path, allow_pickle=True)
    indices_year = np.array([i for i, t in enumerate(times_all) if t.year == year], dtype=np.int64)
    times_year = times_all[indices_year]
    return times_year, indices_year


def pick_samples(times_year, max_samples: int, stride: int, month=None):
    selected = []
    valid_count = 0

    for local_idx, ts in enumerate(times_year):
        if month is not None and int(ts.month) != int(month):
            continue

        try:
            sun_x, sun_y, sun_mask = sun_position(ts)
        except Exception:
            continue

        if valid_count % max(1, stride) == 0:
            selected.append(
                {
                    "local_idx": local_idx,
                    "timestamp": ts,
                    # sun_position() returns image-index style coordinates.
                    # Treat them as (row, col), then plot with x=col, y=row.
                    "sun_row": float(sun_x),
                    "sun_col": float(sun_y),
                }
            )
            if len(selected) >= max_samples:
                break

        valid_count += 1

    return selected


def draw_single_overlay(image, sample, out_path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(image)
    ax.scatter([sample["sun_col"]], [sample["sun_row"]], c="red", s=36, marker="o")
    ax.set_title(str(sample["timestamp"]))
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_contact_sheet(images_with_meta, out_path):
    n = len(images_with_meta)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=120)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        image, sample = images_with_meta[i]
        ax.imshow(image)
        ax.scatter([sample["sun_col"]], [sample["sun_row"]], c="red", s=28, marker="o")
        ax.set_title(str(sample["timestamp"]), fontsize=9)
        ax.set_xlim(0, image.shape[1] - 1)
        ax.set_ylim(image.shape[0] - 1, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    base_out_dir = Path(CONFIG["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    times_year, indices_year = load_times_and_indices(CONFIG["data_dir"], CONFIG["split"], CONFIG["year"])
    if len(times_year) == 0:
        raise RuntimeError("No timestamps available for selected year and split.")

    months = CONFIG.get("months", [None])

    with h5py.File(CONFIG["hdf5_path"], "r") as f:
        images_ds = f[f"{CONFIG['split']}/images_log"]

        for month in months:
            out_dir = base_out_dir / f"month_{int(month):02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            selected = pick_samples(
                times_year,
                max_samples=CONFIG["num_samples"],
                stride=CONFIG["stride"],
                month=month,
            )

            if len(selected) == 0:
                print(f"month={int(month):02d} | no samples selected")
                continue

            images_with_meta = []
            for idx, sample in enumerate(selected):
                global_idx = int(indices_year[sample["local_idx"]])
                image = np.asarray(images_ds[global_idx])

                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)

                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)

                images_with_meta.append((image, sample))

                ts_text = str(sample["timestamp"]).replace(":", "-").replace(" ", "_")
                draw_single_overlay(image, sample, out_dir / f"overlay_{idx:02d}_{ts_text}.png")

            draw_contact_sheet(images_with_meta, out_dir / "contact_sheet.png")
            print(
                f"month={int(month):02d} | saved_single_images={len(images_with_meta)} | "
                f"saved_contact_sheet={out_dir / 'contact_sheet.png'}"
            )

    print("=== Reference Sun Method Visualization Done ===")
    print(f"output_root={base_out_dir}")


if __name__ == "__main__":
    main()
