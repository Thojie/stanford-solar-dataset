#!/usr/bin/env python3
"""
V5 H5 文件检查脚本
检查 output_folder_v5/ 下的所有 H5 输出文件：
1. 图像 H5: 2019_dataset_dual_V5.h5
2. 光流 H5: 2019_dataset_flow_V5.h5
3. 交叉校验两者帧数是否一致
4. times_log 帧间隔统计（确认是否为稳定的 60 秒）
"""

import os
import h5py
import numpy as np
from datetime import datetime


def fmt_count(n):
    """将数字格式化为带逗号分隔的字符串"""
    return f"{n:,}"


def fmt_shape(shape):
    """格式化 shape 为可读字符串"""
    return f"({', '.join(str(s) if s else '?' for s in shape)})"


def print_separator(title="", char="=", width=70):
    """打印分隔线"""
    if title:
        print(f"\n{char * (len(title) + 4)}")
        print(f"  {title}  ")
        print(f"{char * (len(title) + 4)}")
    else:
        print(char * width)


def inspect_image_h5(h5_path):
    """检查图像 H5 文件"""
    print_separator("2019_dataset_dual_V5.h5")
    
    if not os.path.exists(h5_path):
        print(f"❌ 文件不存在: {h5_path}")
        return {}

    with h5py.File(h5_path, 'r') as f:
        groups = ['trainval', 'test']
        result = {}
        
        for grp_name in groups:
            if grp_name not in f:
                print(f"⚠️  缺少组: {grp_name}")
                continue
            
            grp = f[grp_name]
            n_frames = grp['pv_log'].shape[0]
            result[grp_name] = n_frames
            
            print(f"\n┌ {grp_name}: 样本数 {fmt_count(n_frames)}")
            
            # 列出所有数据集
            for dset_name in grp.keys():
                ds = grp[dset_name]
                dtype_str = str(ds.dtype)
                shape_str = fmt_shape(ds.shape)
                chk = ds.chunks
                comp = ds.compression or "none"
                shuffle = "yes" if ds.shuffle else "no"
                chk_str = fmt_shape(chk) if chk else "无"
                print(f"│  {dset_name:25s}  {shape_str:30s}  {dtype_str:8s}  "
                      f"chunks={chk_str}  {comp}+shuffle={shuffle}")
            
            # times_log 帧间隔统计 (仅 trainval 和 test 都展示)
            if 'times_log' in grp:
                times = grp['times_log'][:]
                if len(times) > 1:
                    diffs = np.diff(times)
                    n_faults = np.sum(diffs > 120)
                    print(f"│")
                    print(f"│  📊 帧间隔统计 (秒):")
                    print(f"│     范围:  {diffs.min():.1f} ~ {diffs.max():.1f}")
                    print(f"│     均值:  {diffs.mean():.1f}   中位数: {np.median(diffs):.1f}")
                    print(f"│     标准差: {diffs.std():.1f}")
                    print(f"│     断层 (>120s): {fmt_count(int(n_faults))} 处")
                    
                    # 时间覆盖范围
                    first_dt = datetime.fromtimestamp(times[0])
                    last_dt = datetime.fromtimestamp(times[-1])
                    print(f"│     时间跨度: {first_dt.strftime('%Y-%m-%d %H:%M')} → "
                          f"{last_dt.strftime('%Y-%m-%d %H:%M')}")
                    print(f"│     跨越天数: {(times[-1] - times[0]) / 86400:.1f} 天")
        
        return result


def inspect_flow_h5(h5_path):
    """检查光流 H5 文件"""
    print_separator("2019_dataset_flow_V5.h5")
    
    if not os.path.exists(h5_path):
        print(f"❌ 文件不存在: {h5_path}")
        return {}

    with h5py.File(h5_path, 'r') as f:
        groups = ['trainval', 'test']
        result = {}
        
        for grp_name in groups:
            if grp_name not in f:
                print(f"⚠️  缺少组: {grp_name}")
                continue
            
            grp = f[grp_name]
            for dset_name in grp.keys():
                ds = grp[dset_name]
                result[grp_name] = ds.shape[0]
                dtype_str = str(ds.dtype)
                shape_str = fmt_shape(ds.shape)
                chk = ds.chunks
                comp = ds.compression or "none"
                shuffle = "yes" if ds.shuffle else "no"
                chk_str = fmt_shape(chk) if chk else "无"
                print(f"┌ {grp_name:8s}  {dset_name:25s}  {shape_str:30s}  "
                      f"{dtype_str:8s}  chunks={chk_str}  {comp}+shuffle={shuffle}")
        
        return result


def cross_validate(img_counts, flow_counts):
    """交叉校验图像 H5 和光流 H5 的帧数是否一致"""
    print_separator("交叉校验")
    
    all_match = True
    for grp_name in ['trainval', 'test']:
        img_n = img_counts.get(grp_name, -1)
        flow_n = flow_counts.get(grp_name, -1)
        
        if img_n == -1:
            print(f"❌ {grp_name}: 图像 H5 中缺失")
            all_match = False
        elif flow_n == -1:
            print(f"❌ {grp_name}: 光流 H5 中缺失")
            all_match = False
        elif img_n == flow_n:
            print(f"✅ {grp_name}: 图像 {fmt_count(img_n)} 帧  ==  光流 {fmt_count(flow_n)} 帧  ✓")
        else:
            print(f"❌ {grp_name}: 图像 {fmt_count(img_n)} 帧  !=  光流 {fmt_count(flow_n)} 帧  ✗")
            all_match = False
    
    # 8:2 比例校验
    print()
    total_img = img_counts.get('trainval', 0) + img_counts.get('test', 0)
    total_flow = flow_counts.get('trainval', 0) + flow_counts.get('test', 0)
    
    if total_img > 0:
        train_ratio = img_counts.get('trainval', 0) / total_img * 100
        test_ratio = img_counts.get('test', 0) / total_img * 100
        print(f"图像 H5 分割: trainval {train_ratio:.1f}%  vs  test {test_ratio:.1f}%")
    
    if all_match:
        print(f"\n🎉 全部校验通过！两个 H5 文件完全一致。")
    else:
        print(f"\n⚠️  存在不匹配项，请检查管线。")
    
    return all_match


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(project_root, 'output_folder_v5')
    
    if not os.path.exists(output_folder):
        print(f"❌ 输出目录不存在: {output_folder}")
        print("请先运行 V5 管线 (auto_run.py) 生成数据。")
        return
    
    # 文件路径
    img_h5 = os.path.join(output_folder, '2019_dataset_dual_V5.h5')
    flow_h5 = os.path.join(output_folder, '2019_dataset_flow_V5.h5')
    
    # 检查图像 H5
    img_counts = inspect_image_h5(img_h5)
    
    # 检查光流 H5
    flow_counts = inspect_flow_h5(flow_h5)
    
    # 交叉校验
    if img_counts and flow_counts:
        cross_validate(img_counts, flow_counts)
    
    print("\n" + "=" * 70)
    print("  检查完毕！")
    print("=" * 70)


if __name__ == '__main__':
    main()
