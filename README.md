# Stanford 太阳能数据集预处理工具

## 项目简介

本项目包含对斯坦福站点原始数据（.tar 图像包 与 PV 原始 CSV）的清洗、配准与格式化流程，目标产出可直接用于深度学习训练的 HDF5 数据集与对应时间索引文件。实现细节严格依据仓库内的三个处理文件：`data_process_pv.ipynb`（PV 数据处理）、`new_data_process_image.ipynb`（图像与 HDF5 构建）、`new_data_process_flow.ipynb`（基于 RAFT 的光流生成）。

## 输出文件（示例）
- `output_folder_v3/2019_dataset_dual_V3.h5`：主图像数据集（`trainval` 组，内含 `global_images_log`, `local_images_log`, `pv_log`, `sun_pos`）。
- `output_folder_v3/2019_dataset_flow_V3.h5`：对应的光流数据集（`trainval/global_flow_log`，dtype=float16，shape=(N,2,256,256)，第 0 帧为占位零帧）。
- `output_folder_v3/times_trainval.npy` 与 `times_test.npy`：严格的物理时间轴索引（按 8:2 划分，保存为 numpy datetime64/py datetime 序列）。
- `pv_data/pv_output_valid.pkl`：PV 处理后的 10s 采样并筛选后的 Pandas Series（频率为 10s）。

## 处理流程总览（严格对应代码实现）

1. PV 数据清洗（`data_process_pv.ipynb`）
   - 读取 `pv_data/2019_pv_raw.csv`，解析时间列为 `datetime`，并以时间作为索引。
   - 使用 `scipy.interpolate.interp1d` 对原始观测数据做逐秒插值，随后以 10s 重采样并取均值，得到每 10s 一个值的时间序列。
   - 人工/可视化筛查：通过逐日绘图检查并列出 `invalid_dates`（手动黑名单段），对间隔超过 1 小时的原始记录段标记为缺失（不可信的强制插值段也会被标记）。
   - 最终筛选条件：去除黑名单段、去除长缺失段产生的插值（>1h），并过滤夜间或零输出（仅保留 `pv_output > 0`）。
   - 保存：`pv_output_valid.pkl`（Pandas Series，index = DatetimeIndex，freq=10s）。

2. 图像解包、校准与 HDF5 构建（`new_data_process_image.ipynb`）
   - 配置：`tar_folder` 指向原始 `.tar` 图像集合，`output_h5_path` 指向输出 HDF5（默认写入 `output_folder_v3/2019_dataset_dual_V3.h5`）。
   - 太阳位置与镜头畸变校准：使用理论太阳位置计算函数 `get_theo_sun_position()`（基于天文几何推算）为基准，在若干手动指定的晴天标定日 `sunny_calibration_days` 中用 CV 检测日斑（阈值、最小轮廓面积判断）建立 `MONTHLY_INTERP_TABLE`（每月的时间点与 dx/dy 偏移）。对缺失月份使用 `FALLBACK_MONTHS` 回退到相邻月份的数据。
   - `get_smart_sun_position()`：基于当日时间插值月度偏移表得到鲁棒的太阳像素坐标（理论坐标 + 插值偏移）。
   - 双分支图像采样：`extract_dual_branch_images(img, sun_x, sun_y)` 返回：
     - `global_images_log`：RGB、缩放到 256x256，通道顺序转换为 C,H,W（dtype uint8）。
     - `local_images_log`：以太阳为中心做大裁剪（HUGE_CROP=768）填充至画布，缩放到 128x128，RGB，C,H,W。
   - 内存缓冲写入：使用 `BUFFER_SIZE = 100` 将读取的帧分批写入 HDF5（避免频繁 resize 造成碎片），HDF5 结构如下（`trainval` 组）：
     - `global_images_log`: shape=(0,3,256,256), dtype=uint8, chunks=(BUFFER_SIZE,3,256,256)
     - `local_images_log`: shape=(0,3,128,128), dtype=uint8, chunks=(BUFFER_SIZE,3,128,128)
     - `pv_log`: shape=(0,), dtype=float32, chunks=(BUFFER_SIZE,)
     - `sun_pos`: shape=(0,2), dtype=int32, chunks=(BUFFER_SIZE,2)
   - 每次缓冲满后调用 `flush_buffer()` 做一次批量写入；完成后会生成 `all_recorded_times`（按记录顺序）并按 80%/20% 划分为 `times_trainval.npy` 与 `times_test.npy`。

3. 光流生成（`new_data_process_flow.ipynb`）
   - 模型加载：使用仓库内 `RAFT` 模型实现，权重路径为 `RAFT/models/raft-small.pth`。加载时处理 `state_dict` 的 `module.` 前缀并支持 `weights_only` 加载。
   - 降噪配置 `DENOISE_CFG`：
     - `median_size`: 3
     - `tau_abs`: 0.6
     - `tau_rel`: 0.08
     - `min_component_area`: 64
   - `hard_denoise_flow(flow_np, valid_mask)`：对 u、v 做中值滤波，计算速度幅值并使用阈值 tau = max(tau_abs, tau_rel * max_mag) 保留大于阈值连通分量，移除面积小于 `min_component_area` 的小连通域。
   - `build_valid_sky_mask(image1, image2)`：基于中心圆（radius_ratio=0.49）与像素黑度阈值（black_threshold=8）联合判断有效天区。
   - 生成流程 `generate_flow_h5_v3(source_h5, times_dir, output_h5)`：
     - 加载 `times_trainval.npy` 与 `times_test.npy` 并拼接为 `all_times`（用于时间连续性检查）。
     - 读取 `source_h5['trainval']['global_images_log']` 并断言帧数与时间戳一致。
     - 在磁盘上预分配 `global_flow_log` 数据集：shape `(total_frames, 2, 256, 256)`, dtype `float16`, chunks `(1,2,256,256)`, compression `lzf`。
     - 写入规则：第 0 帧写入全零占位；其余帧按 `buffer_size = 500` 分批处理：
       - 每批次从源 HDF5 读取（含上一帧做差分），对每对连续帧调用 RAFT（`iters=20, test_mode=True`）计算光流。
       - 若相邻时间戳差 `> 65s`，则将该帧的光流写成全零（视为不可用）。
       - 对得到的光流做 `hard_denoise_flow`（结合 `build_valid_sky_mask`），并转换为 `(2,256,256)`、`float16` 写入目标数据集。

## HDF5 生成方法（实现细节与代码位置）

    1) 图像 HDF5（实现：`new_data_process_image.ipynb`）
      - 创建：在 `with h5py.File(output_h5_path, 'w') as h5f:` 内通过 `grp.create_dataset` 建表，初始 `shape=(0, ...)`、`maxshape=(None, ...)` 使表可追加，使用 `chunks=(BUFFER_SIZE, ...)` 与 `compression='lzf'`。
      - 写入：使用 `buffers`（`BUFFER_SIZE = 100`）在内存中累积 `global`, `local`, `pv`, `sun` 四类数据；`flush_buffer(grp)` 内先用 `grp[key].resize(curr_len + n_new, axis=0)` 扩容，再通过切片一次性写入（例如 `grp['global_images_log'][curr_len:] = np.stack(buffers['global'])`）。
      - 优点：批量写入显著降低 IO、避免 HDF5 碎片化，并保持磁盘布局连续。

    2) 光流 HDF5（实现：`new_data_process_flow.ipynb`）
      - 预分配：目标文件在创建时即预分配完整尺寸 `flow_ds = dst_grp.create_dataset('global_flow_log', shape=(total_frames, 2, 256, 256), dtype='float16', chunks=(1,2,256,256), compression='lzf')`，保证按索引可直接写入。
      - 写入策略：第 0 帧写入全零占位；其余按 `buffer_size = 500` 分批处理，计算得到的 `out_flows` 使用切片写回：`flow_ds[start_idx:end_idx] = np.stack(out_flows)`。
      - 时间完整性：处理前检查 `all_times` 的时间差，若相邻帧 `time_delta > 65s` 则对应光流写入全零以示不可用。

    3) PV 数据持久化（实现：`data_process_pv.ipynb`）
      - 结果以 Pandas 序列持久化为 `pv_output_valid.pkl`：`pv_output_valid.to_pickle(os.path.join(output_folder,'pv_output_valid.pkl'))`，供后续图像—PV 对齐使用。

    4) 关键函数与位置索引（便于快速定位）
      - `flush_buffer(grp)`：`new_data_process_image.ipynb`（负责批量扩容与写入）。
      - `extract_dual_branch_images(img, sun_x, sun_y)`：`new_data_process_image.ipynb`（生成 `global_images_log` 与 `local_images_log` 的采样逻辑）。
      - `generate_flow_h5_v3(source_h5, times_dir, output_h5)`：`new_data_process_flow.ipynb`（光流计算、降噪、预分配与写回核心）.


## 运行环境与依赖

- Python 3.8+
- 主要 Python 包（可用 pip 安装）：

```bash
pip install numpy pandas h5py opencv-python tqdm scipy matplotlib torch
```

注：RAFT 模型使用 GPU 时需要相应的 CUDA + PyTorch 版本；仓库中包含 `RAFT` 源代码与 `models/raft-small.pth` 权重。

## 使用方法（示例命令）

1) 生成 PV 处理结果（在 Jupyter 中按单元顺序运行 `data_process_pv.ipynb`），或在命令行执行：

```bash
# 在项目根目录下执行（需要已安装 nbconvert）
jupyter nbconvert --to notebook --execute data_process_pv.ipynb --ExecutePreprocessor.timeout=600
```

运行完成后会在 `pv_data/` 下生成 `pv_output_valid.pkl`。

2) 构建图像 HDF5 数据集：

```bash
jupyter nbconvert --to notebook --execute new_data_process_image.ipynb --ExecutePreprocessor.timeout=0
```

或打开 `new_data_process_image.ipynb` 在 Notebook 中按顺序运行各单元。完成后会生成 `output_folder_v3/2019_dataset_dual_V3.h5` 及 `times_trainval.npy` / `times_test.npy`。

3) 生成光流数据（需先生成图像 HDF5 与时间索引）：

```bash
jupyter nbconvert --to notebook --execute new_data_process_flow.ipynb --ExecutePreprocessor.timeout=0
```

或在 Python 环境中直接运行该 Notebook 中最后的 `if __name__ == '__main__'` 段所调用的 `generate_flow_h5_v3()`（确保 `WEIGHTS` 路径存在且 GPU/CPU 可用）。运行后会生成 `output_folder_v3/2019_dataset_flow_V3.h5`。

## 文件结构（仓库相关项）

- `data_process_pv.ipynb` — PV 插值、去坏值、10s 重采样并输出 `pv_output_valid.pkl`。
- `new_data_process_image.ipynb` — 解压 `.tar`、图像双分支采样、镜头畸变标定与 HDF5 写入（`2019_dataset_dual_V3.h5`）。
- `new_data_process_flow.ipynb` — 基于 RAFT 的批量光流计算与降噪，输出 `2019_dataset_flow_V3.h5`。
- `pv_data/2019_pv_raw.csv` — 原始 PV CSV 源文件。
- `output_folder_v3/` — 处理后产物目录（HDF5、times_*.npy、fisheye_center_radius.*）。

## 注意事项与常见问题

- 路径：所有 Notebook 默认使用项目根目录（`os.getcwd()`）作为 `project_path`，请在正确的工作目录下运行或先修改相应路径变量。
- PV 手动黑名单：`data_process_pv.ipynb` 中 `invalid_dates` 依靠人工/可视化检查生成，若要完全自动化需额外规则。不要去除该步骤以免把坏数据带入训练集。
- 时间匹配：图像到 PV 的匹配使用 Pandas 的索引精确匹配（`find_time_within_pdseries` 使用 `np.searchsorted`）。请确保 `pv_output_valid.pkl` 的时间索引覆盖图像时间点。
- 内存与显存：光流生成使用 RAFT（可在 GPU 上加速），内存/显存受 `buffer_size`（图像批量）与 RAFT 模型大小影响，可适当调整 `buffer_size`（`new_data_process_flow.ipynb` 中默认 500）。
- RAFT 权重：`RAFT/models/raft-small.pth` 必须存在，加载器会尝试兼容多种 state_dict 格式。
- 首帧占位：光流数据集的第 0 帧为全零占位（因光流为帧间差分）。

## 联系与扩展建议

若需把 HDF5 格式调整为其他训练管线的输入格式（例如 TFRecord 或 torchvision Dataset 直接读取结构），可以在 `new_data_process_image.ipynb` 的 `flush_buffer` 基础上新增导出脚本。

----

