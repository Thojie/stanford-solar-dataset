import h5py
import numpy as np

p = r"e:\造数据集\output_folder_v3\2019_dataset_flow_V3.h5"

with h5py.File(p, "r") as f:
    d = f["trainval"]["global_flow_log"]
    n = d.shape[0]

    gmin = np.inf
    gmax = -np.inf
    absmax = 0.0

    step = 256
    mini = (-1, -1, -1, -1)
    maxi = (-1, -1, -1, -1)

    for i in range(0, n, step):
        x = d[i : min(i + step, n)]

        lmin = float(np.min(x))
        lmax = float(np.max(x))
        labs = float(np.max(np.abs(x)))

        if lmin < gmin:
            gmin = lmin
            idx = np.argwhere(x == lmin)[0]
            mini = (i + int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))

        if lmax > gmax:
            gmax = lmax
            idx = np.argwhere(x == lmax)[0]
            maxi = (i + int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))

        if labs > absmax:
            absmax = labs

    print("shape", d.shape, "dtype", d.dtype)
    print("global_min", gmin, "at", mini)
    print("global_max", gmax, "at", maxi)
    print("global_abs_max", absmax)
