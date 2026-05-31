# V4 vs V5 差异对照文档

## 📂 文件结构对比

| 用途 | V4 文件 | V5 文件 |
|------|---------|---------|
| PV 数据清洗 | `v4_dataprocess/data_process_pv.ipynb` | `v5_dataprocess/v5_data_process_pv.py` |
| 图像数据集生成 | `v4_dataprocess/data_process_image.ipynb` | `v5_dataprocess/v5_data_process_image.py` |
| 光流计算 | `v4_dataprocess/data_process_flow.ipynb` | `v5_dataprocess/v5_data_process_flow.py` |
| 外部切分脚本 | `v4_dataprocess/image_group_to_trainval_and_test.py` | ❌ **已内聚到 v5 Image 脚本末尾** |
| 输出目录 | `output_folder_v4/` | `output_folder_v5/` |

---

## 1️⃣ PV 数据清洗脚本差异

| 对比项 | V4 (`data_process_pv.ipynb`) | V5 (`v5_data_process_pv.py`) |
|--------|------------------------------|------------------------------|
| **文件格式** | `.ipynb` (Jupyter Notebook) | `.py` (标准 Python 脚本) |
| **夜间过滤阈值** | `pv_output_10s_int > 0` | 🔴 `pv_output_10s_int > 0.05` (提升至 50W) |
| **输出文件名** | `pv_output_valid.pkl` | 🔴 `V5_pv_valid.pkl` |

---

## 2️⃣ 图像数据集生成脚本差异（核心改造）

| 对比项 | V4 (`data_process_image.ipynb`) | V5 (`v5_data_process_image.py`) |
|--------|---------------------------------|---------------------------------|
| **文件格式** | `.ipynb` (Jupyter Notebook) | `.py` (标准 Python 脚本) |
| **PV 数据来源** | `pv_data/pv_output_valid.pkl` | 🔴 `pv_data/V5_pv_valid.pkl` |
| **时间过滤方式** | `if curr_time.hour < 7 or ...` **硬编码** | 🔴 **废除!** 改为 `if pv_val < 0.05: continue` 动态物理阈值 |
| **NaN/异常处理** | 无显式处理 | 🔴 `try/except (ValueError, IndexError): continue` 保护 |
| **H5 写入策略** | 直接写入最终 H5 | 🔴 **两步走**: ① 暂存临时 H5 → ② 末尾重构切分 |
| **8:2 切分位置** | 外部脚本 `image_group_to_trainval_and_test.py` | 🔴 **内聚**到脚本末尾的 `repackage_to_v5()` 函数 |
| **H5 组结构** | 仅 `trainval` 组（需要外部脚本补 `test`） | 🔴 **一次性生成** `trainval` + `test` 双组 |
| **临时文件** | 无 | 🔴 `2019_dataset_dual_V5_temp.h5`（切分后自动删除） |
| **I/O 压缩锁定** | 已有 `compression='lzf'`，`chunks` 自动 | 🔴 **强制硬编码**: `chunks=100`，图像 `shuffle=True` |
| **输出文件名** | `2019_dataset_dual_V4.h5` | 🔴 `2019_dataset_dual_V5.h5` |

### 🔴 V5 新增/改造代码段示意

**改造 A: 废除 hour 硬编码**
```python
# V4 (已删除):
# if curr_time.hour < 7 or (curr_time.hour == 7 and curr_time.minute < 30) or \
#    curr_time.hour > 17 or (curr_time.hour == 17 and curr_time.minute > 30): continue

# V5 (新逻辑):
try:
    pv_val = float(pv_output_all.iloc[pv_idx])
except (ValueError, IndexError):
    continue  # 处理 NaN 或索引异常

if pv_val < 0.05:
    continue  # PV 功率过低，跳过该帧
```

**改造 B: 内聚重构函数 `repackage_to_v5()`**
```python
def repackage_to_v5(temp_path, final_path, split_idx):
    # 1. 读取临时 H5
    # 2. 按时间顺序 8:2 切分为 trainval / test
    # 3. 强制 I/O 参数: compression='lzf', chunks=100, shuffle=True(图像)
    # 4. 验证总帧数一致后删除临时文件
```

**改造 C: 强制 Chunk 硬编码**
```python
V5_CHUNK_MAP = {
    'global_images_log': (100, 3, 256, 256),
    'local_images_log': (100, 3, 128, 128),
    'pv_log': (100,),
    'sun_pos': (100, 2)
}
```

---

## 3️⃣ 光流计算脚本差异

| 对比项 | V4 (`data_process_flow.ipynb`) | V5 (`v5_data_process_flow.py`) |
|--------|--------------------------------|--------------------------------|
| **文件格式** | `.ipynb` (Jupyter Notebook) | `.py` (标准 Python 脚本) |
| **读取源 H5** | `2019_dataset_dual_V4.h5` | 🔴 `2019_dataset_dual_V5.h5` |
| **遍历组** | 仅 `trainval` | 🔴 **遍历双组**: `trainval` + `test` |
| **时间轴校验** | 只有训练集时间轴 | 🔴 **各组使用独立时间轴** |
| **输出文件名** | `2019_dataset_flow_V4.h5` | 🔴 `2019_dataset_flow_V5.h5` |
| **Chunk 配置** | `(100, 2, 256, 256)` | ✅ 维持不变 + `shuffle=True` |
| **模型降噪逻辑** | 完整保留 | ✅ 完全迁移 |

---

## 4️⃣ 额外差异

| 对比项 | V4 | V5 |
|--------|----|----|
| **文件夹** | `v4_dataprocess/` | `v5_dataprocess/` |
| **输出目录** | `output_folder_v4/` | `output_folder_v5/` |
| **外部依赖脚本** | `image_group_to_trainval_and_test.py`（独立文件） | ❌ **无外部依赖**（逻辑内聚到 Image 脚本） |

---

## 🔴 所有修改点汇总（共 10 项）

1. ✅ PV 阈值: `>0` → `>0.05`
2. ✅ PV 输出: `pv_output_valid.pkl` → `V5_pv_valid.pkl`
3. ✅ Image: 废除 `hour` 硬编码过滤
4. ✅ Image: 替换为 `PV<0.05` 动态物理阈值
5. ✅ Image: 新增 NaN/索引异常处理
6. ✅ Image: 临时 H5 → 8:2 切分 → 删除临时文件 内聚流水线
7. ✅ Image: 强制 `chunks=100` 硬编码
8. ✅ Image: 图像数据集强制 `shuffle=True`
9. ✅ Flow: 切换读取源为 V5 H5
10. ✅ Flow: 遍历 `trainval` + `test` 双组
