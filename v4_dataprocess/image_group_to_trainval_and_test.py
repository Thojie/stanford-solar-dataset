import os
import h5py
import numpy as np

def repackage_to_perfect_v4(v3_path, v4_path, split_idx):
    print(f"\n🔄 正在执行完美物理重构: {os.path.basename(v3_path)} -> {os.path.basename(v4_path)}")
    
    # 🌟 强制硬编码，杜绝 HDF5 自作聪明！
    CHUNK_MAP = {
        'global_images_log': (100, 3, 256, 256),
        'local_images_log':  (100, 3, 128, 128),
        'pv_log':            (100,),
        'sun_pos':           (100, 2)
    }
    
    with h5py.File(v3_path, 'r') as src, h5py.File(v4_path, 'w') as dst:
        grp_src = src['trainval']
        
        # 创建标准的 trainval 和 test
        grp_train = dst.create_group('trainval')
        grp_test = dst.create_group('test')
        
        for key in grp_src.keys():
            dataset = grp_src[key]
            total_len = dataset.shape[0]
            my_chunk = CHUNK_MAP.get(key, True)
            
            print(f"  ✂️ 正在切分并重组 [{key}]，总长度: {total_len} ...")
            
            # --- 写入 Trainval 集 ---
            grp_train.create_dataset(
                key, 
                data=dataset[:split_idx], 
                compression='lzf',
                shuffle=True if 'images' in key else False, # 🌟 图像加 shuffle，提速提压缩率
                chunks=my_chunk
            )
            
            # --- 写入 Test 集 ---
            grp_test.create_dataset(
                key, 
                data=dataset[split_idx:], 
                compression='lzf',
                shuffle=True if 'images' in key else False,
                chunks=my_chunk
            )
            
    print(f"✅ {os.path.basename(v4_path)} 重构完成！Trainval/Test 绝对物理隔离，Chunk 完全对齐！")

if __name__ == '__main__':
    dsrc_dir = r"E:\造数据集\output_folder_v3"
    output_dir = r"E:\造数据集\output_folder_v4"
    os.makedirs(output_dir, exist_ok=True)
    # 获取切分点
    times_train = np.load(os.path.join(dsrc_dir, 'times_trainval.npy'), allow_pickle=True)
    split_idx = len(times_train)
    
    # 重构图像数据集
    dual_v3 = os.path.join(dsrc_dir, '2019_dataset_dual_V3.h5')
    dual_v4 = os.path.join(output_dir, '2019_dataset_dual_V4.h5')
    
    repackage_to_perfect_v4(dual_v3, dual_v4, split_idx)