import h5py
import numpy as np

def check_files(path_a, path_b):
    with h5py.File(path_a, 'r') as f1, h5py.File(path_b, 'r') as f2:
        # 简单比对一下数据集形状和分块
        for key in f1['trainval'].keys():
            shape1 = f1['trainval'][key].shape
            shape2 = f2['trainval'][key].shape
            chunk1 = f1['trainval'][key].chunks
            chunk2 = f2['trainval'][key].chunks
            
            if shape1 != shape2 or chunk1 != chunk2:
                print(f"❌ 发现差异！Key: {key}")
                return False
        print("✅ 两个文件物理结构完全一致！")
        return True

# 改成你本地那两个文件的路径
check_files(r"E:\造数据集\output_folder_v3\2019_dataset_dual_V4.h5", r"E:\造数据集\output_folder_v4\2019_dataset_dual_V4.h5")

import h5py

def inspect_structure(file_path):
    structure = {}
    with h5py.File(file_path, 'r') as f:
        for key in f['trainval'].keys():
            dset = f['trainval'][key]
            # 获取物理存储的核心参数
            structure[key] = {
                'shape': dset.shape,
                'chunks': dset.chunks,
                'compression': dset.compression,
                'addr': dset.id.get_offset() # 这是数据在硬盘上的物理起始地址
            }
    return structure

path_a = r"E:\造数据集\output_folder_v3\2019_dataset_dual_V4.h5"
path_b = r"E:\造数据集\output_folder_v4\2019_dataset_dual_V4.h5"

struct_a = inspect_structure(path_a)
struct_b = inspect_structure(path_b)

print(f"{'Dataset Key':<25} | {'Shape Diff':<15} | {'Chunk Diff':<15} | {'Addr Diff'}")
print("-" * 80)

for key in struct_a:
    a = struct_a[key]
    b = struct_b[key]
    
    shape_diff = "不同" if a['shape'] != b['shape'] else "相同"
    chunk_diff = "不同" if a['chunks'] != b['chunks'] else "相同"
    addr_diff = "不同" if a['addr'] != b['addr'] else "相同"
    
    print(f"{key:<25} | {shape_diff:<15} | {chunk_diff:<15} | {addr_diff}")

import h5py
import time

def test_speed(file_path):
    with h5py.File(file_path, 'r') as f:
        dset = f['test']['global_images_log']
        print(f"[{file_path}]")
        print(f" -> Chunk: {dset.chunks}")
        
        # 模拟你在代码里用的读取：一次性连续读 15 帧
        t0 = time.time()
        _ = dset[1000:1015]
        t1 = time.time()
        print(f" -> 15帧读取耗时: {(t1 - t0) * 1000:.2f} ms")
        print("-" * 30)

# 把你的两个文件名填进去
test_speed(r"E:\造数据集\output_folder_v3\2019_dataset_dual_V4.h5")
test_speed(r"E:\造数据集\output_folder_v4\2019_dataset_dual_V4.h5")