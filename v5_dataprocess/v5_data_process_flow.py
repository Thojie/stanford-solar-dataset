"""
V5 Optical Flow Generation Script
===============================================
核心改造 (对比 V4):
  1. 🔴 读取源改为 2019_dataset_dual_V5.h5
  2. 🔴 完美适配双组结构，分别遍历 trainval 和 test
  3. Chunk 维持 (100, 2, 256, 256), shuffle=True
  4. 🔴 时间戳从 H5 内部 times_log 读取 (无需外部 .npy)

输入: output_folder_v5/2019_dataset_dual_V5.h5 (包含 trainval/times_log + test/times_log)
输出: output_folder_v5/2019_dataset_flow_V5.h5 (双组: trainval + test)
"""

import h5py
import numpy as np
import torch
import os
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage as ndi
import datetime as dt  # 🔴 V5: 后续 datetime 工具保留
import sys
sys.path.append(os.getcwd())
from RAFT.core.raft import RAFT


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


ROOT = Path.cwd()
if not (ROOT / "RAFT").exists() and (ROOT.parent / "RAFT").exists():
    ROOT = ROOT.parent

WEIGHTS = ROOT / "RAFT" / "models" / "raft-small.pth"

# 降噪配置 (与 V4 完全一致)
DENOISE_CFG = {
    "median_size": 3,
    "tau_abs": 0.6,
    "tau_rel": 0.08,
    "min_component_area": 64,
}

# ==========================================
# 1. 核心模型与算法函数 (与 V4 完全一致)
# ==========================================

def load_raft_model(weight_path=WEIGHTS, small=True):
    """加载 RAFT 光流模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = DotDict(small=small, mixed_precision=True, alternate_corr=False)
    model = RAFT(args).to(device).eval()

    if not weight_path.exists():
        raise FileNotFoundError(f"找不到 RAFT 权重: {weight_path}")

    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        first_key = next(iter(state))
        if first_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    return model, device


def hard_denoise_flow(flow_np, valid_mask, cfg=DENOISE_CFG):
    """硬降噪: 中值滤波 + 幅度阈值 + 小连通域剔除"""
    u, v = flow_np[..., 0], flow_np[..., 1]
    u_med = ndi.median_filter(u, size=cfg["median_size"])
    v_med = ndi.median_filter(v, size=cfg["median_size"])
    mag = np.sqrt(u_med ** 2 + v_med ** 2)

    max_mag = float(mag[valid_mask].max()) if valid_mask.any() else float(mag.max())
    tau = max(cfg["tau_abs"], cfg["tau_rel"] * max_mag)

    keep = valid_mask & (mag >= tau)
    labeled, num = ndi.label(keep)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        small_labels = np.where(counts < cfg["min_component_area"])[0]
        if len(small_labels) > 0:
            keep[np.isin(labeled, small_labels)] = False

    out = np.zeros_like(flow_np, dtype=np.float32)
    out[..., 0][keep] = u_med[keep]
    out[..., 1][keep] = v_med[keep]
    return out, keep, tau


def build_valid_sky_mask(image1_np, image2_np, radius_ratio=0.49, black_threshold=8):
    """构建有效天空区域掩码（圆形+非黑像素）"""
    h, w = image1_np.shape[:2]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * radius_ratio)
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

    valid1 = np.any(image1_np > black_threshold, axis=2)
    valid2 = np.any(image2_np > black_threshold, axis=2)
    return circle_mask & valid1 & valid2


# ==========================================
# 2. 🔴 V5 光流生成引擎 (双组遍历)
# ==========================================

BUFFER_SIZE = 100


def generate_flow_h5_v5(source_h5, output_h5):
    """
    🔴 V5 光流生成引擎:
      - 遍历 trainval 和 test 双组
      - 从 H5 内部 times_log 读取时间戳 (Unix 秒, float64)
      - 使用原始 Unix 秒差进行时间断层校验
      - 输出双组结构的 flow H5
    """
    print("=" * 60)
    print("🚀 V5 Optical Flow Generation Engine")
    print("=" * 60)

    print("\n⏳ 正在挂载 RAFT 模型...")
    model, device = load_raft_model(small=True)
    print(f"   ✅ 模型已加载至 {device}")

    with h5py.File(source_h5, 'r') as src, h5py.File(output_h5, 'w') as dst:
        # 🔴 遍历双组: 先 trainval，后 test
        for grp_name in ['trainval', 'test']:
            # ✅ 从 H5 内部读取时间戳（Unix 秒, float64）
            times_array = src[grp_name]['times_log'][:]
            total_frames = len(times_array)
            print(f"\n{'=' * 50}")
            print(f"🎬 开始处理 [{grp_name}] 组，共计 {total_frames} 帧")
            print(f"{'=' * 50}")

            src_imgs = src[grp_name]['global_images_log']
            assert src_imgs.shape[0] == total_frames, \
                f"🚨 [{grp_name}] 图像数({src_imgs.shape[0]})与时间戳数({total_frames})不匹配!"

            # 预分配当前组的光流存储空间
            dst_grp = dst.create_group(grp_name)
            flow_ds = dst_grp.create_dataset(
                'global_flow_log',
                shape=(total_frames, 2, 256, 256),
                dtype='float16',
                chunks=(100, 2, 256, 256),  # 黄金分块
                compression='lzf',
                shuffle=True                # float16 极致压缩
            )
            print(f"   ✅ 已为 [{grp_name}] 预分配存储空间: {flow_ds.shape}")

            # 第 0 帧占位 (无前帧可计算光流)
            flow_ds[0] = np.zeros((2, 256, 256), dtype=np.float16)

            raft_batch_size = 500

            for start_idx in tqdm(
                range(1, total_frames, raft_batch_size),
                desc=f"🚀 [{grp_name}] 光流引擎全速推进"
            ):
                end_idx = min(start_idx + raft_batch_size, total_frames)

                # 批量读取图像
                img_buffer = src_imgs[start_idx - 1: end_idx]

                # 预分配批次输出
                batch_out = np.zeros(
                    (end_idx - start_idx, 2, 256, 256), dtype=np.float16
                )

                for local_i in range(end_idx - start_idx):
                    global_i = start_idx + local_i

                    # 🔴 V5: times_log 存储为 Unix 秒 (float64)，直接用差值比较
                    time_delta = times_array[global_i] - times_array[global_i - 1]
                    if time_delta > 65.0:
                        # 时间断层 > 65s，第 0 帧占位（已经初始化为 0）
                        continue

                    # 极速取图: img1 为 t-1 帧，img2 为 t 帧
                    img1 = img_buffer[local_i]      # t-1
                    img2 = img_buffer[local_i + 1]  # t

                    t1 = torch.from_numpy(img1).float().unsqueeze(0).to(device)
                    t2 = torch.from_numpy(img2).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        _, flow_up = model(t1, t2, iters=20, test_mode=True)
                        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()

                    # 降噪处理
                    img1_np = img1.transpose(1, 2, 0)
                    img2_np = img2.transpose(1, 2, 0)
                    valid_mask = build_valid_sky_mask(img1_np, img2_np)

                    flow_dn_raw, _, _ = hard_denoise_flow(flow_np, valid_mask)
                    flow_dn = flow_dn_raw.transpose(2, 0, 1).astype(np.float16)

                    batch_out[local_i] = flow_dn

                # 批量写入硬盘
                flow_ds[start_idx: end_idx] = batch_out

            # 刷入硬盘
            dst.flush()
            print(f"   ✅ [{grp_name}] 组光流计算完成")

    # 最终验证
    print(f"\n{'=' * 60}")
    print("📊 最终文件验证:")
    with h5py.File(output_h5, 'r') as f_verify:
        for grp_name in ['trainval', 'test']:
            grp = f_verify[grp_name]
            ds = grp['global_flow_log']
            print(f"   [{grp_name}] flow shape: {ds.shape}, dtype: {ds.dtype}")
            print(f"       chunks: {ds.chunks}, compression: {ds.compression}, shuffle: {ds.shuffle}")

    print(f"\n🎉 V5 光流数据生成完毕!")
    print(f"📁 输出文件: {output_h5}")
    print(f"{'=' * 60}")


# ==========================================
# 3. 启动入口
# ==========================================
if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'output_folder_v5')
    src_h5 = os.path.join(data_dir, '2019_dataset_dual_V5.h5')
    out_h5 = os.path.join(data_dir, '2019_dataset_flow_V5.h5')

    print("V5 Optical Flow Pipeline")
    print(f"   读取源: {src_h5}")
    print(f"   输出至: {out_h5}")

    if not os.path.exists(src_h5):
        raise FileNotFoundError(
            f"🚨 找不到源文件: {src_h5}\n"
            f"   请先运行 v5_data_process_image.py 生成图像数据集!"
        )

    generate_flow_h5_v5(
        source_h5=src_h5,
        output_h5=out_h5
    )
