from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_pipeline import SkippdDataset


CONFIG = {
    "hdf5_path": "dataset/2017_2019_images_pv_processed.hdf5",
    "year": 2019,
    "data_dir": "dataset",
    "img_history": 5,
    "pv_history": 30,
    "forecast_horizon": 15,
    "num_examples": 8,
    "start_index": 0,
    "local_crop_size": 24,
    "output_dir": "experiments/local_crop_viz",
    "output_name": "local_crop_examples.png",
}


def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train, _, _, _ = SkippdDataset.build_datasets(
        CONFIG["hdf5_path"],
        year=CONFIG["year"],
        data_dir=CONFIG["data_dir"],
        img_history=CONFIG["img_history"],
        pv_history=CONFIG["pv_history"],
        forecast_horizon=CONFIG["forecast_horizon"],
        enable_solar_position=True,
        enable_local_crop=True,
        local_crop_size=CONFIG["local_crop_size"],
        image_mode="local",
    )

    n = CONFIG["num_examples"]
    start = CONFIG["start_index"]
    end = min(len(ds_train), start + n)

    rows = end - start
    if rows <= 0:
        raise RuntimeError("No samples available for visualization.")

    fig, axes = plt.subplots(rows, 2, figsize=(8, 3.6 * rows), dpi=120)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, idx in enumerate(range(start, end)):
        sample = ds_train[idx]

        img_global = sample["x_img_global"][-1].cpu().numpy().transpose(1, 2, 0)
        img_local = sample["x_img_local"][-1].cpu().numpy().transpose(1, 2, 0)
        sun_u = float(sample["sun_xy"][-1, 0].item())
        sun_v = float(sample["sun_xy"][-1, 1].item())

        ax0 = axes[r, 0]
        ax1 = axes[r, 1]

        ax0.imshow(img_global)
        ax0.scatter([sun_u], [sun_v], c="red", s=22)

        half = CONFIG["local_crop_size"] // 2
        x0 = int(np.round(sun_u)) - half
        y0 = int(np.round(sun_v)) - half
        rect = plt.Rectangle((x0, y0), CONFIG["local_crop_size"], CONFIG["local_crop_size"],
                             fill=False, edgecolor="yellow", linewidth=1.2)
        ax0.add_patch(rect)

        ax0.set_title(f"Global idx={idx}")
        ax0.set_xlim(0, img_global.shape[1] - 1)
        ax0.set_ylim(img_global.shape[0] - 1, 0)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.imshow(img_local)
        ax1.set_title("Local crop")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.tight_layout()
    out_path = out_dir / CONFIG["output_name"]
    fig.savefig(out_path)
    plt.close(fig)

    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
