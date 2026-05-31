"""
V5 Image Dataset Generation Script
=============================================
核心改造 (对比 V4):
  1. 🔴 废除 hour 硬编码过滤 → 改为 PV_output < 0.05 动态物理阈值
  2. 🔴 先写入临时 H5 → 脚本末尾自动 8:2 物理隔离切分 → 删除临时文件
  3. I/O 强制锁定: chunks=100, compression='lzf', shuffle=True(图像)

依赖: pv_data/V5_pv_valid.pkl (由 v5_data_process_pv.py 生成)
输出: output_folder_v5/2019_dataset_dual_V5.h5 (trainval + test 双组)
"""

import cv2
import numpy as np
import pandas as pd
import h5py
import tarfile
import os
import datetime as dt
from math import *
from tqdm import tqdm
import calendar

# ==========================================
# 路径配置与全局参数
# ==========================================
project_path = os.getcwd()
tar_folder = os.path.join(project_path, '2019StanFord_image')
pv_data_path = os.path.join(project_path, 'pv_data', 'V5_pv_valid.pkl')

output_folder = os.path.join(project_path, 'output_folder_v5')
os.makedirs(output_folder, exist_ok=True)

# 临时文件: 先将所有合格帧写入此文件
temp_h5_path = os.path.join(output_folder, '2019_dataset_dual_V5_temp.h5')
# 最终文件: 8:2 切分后写入此文件
final_h5_path = os.path.join(output_folder, '2019_dataset_dual_V5.h5')

# 晴天标定日 (用于镜头畸变校准，与 V4 保持一致)
sunny_calibration_days = [
    (2019, 1, 25), (2019, 5, 31), (2019, 6, 23),
    (2019, 7, 14), (2019, 8, 11), (2019, 9, 6), (2019, 10, 14)
]

print(f"✅ V5 配置加载完成")
print(f"   临时 H5: {temp_h5_path}")
print(f"   最终 H5: {final_h5_path}")

# ==========================================
# 🌟 太阳位置计算与插值台账提取 (与 V4 完全一致)
# ==========================================

def doy_tod_conv(date_and_time, longitude=-122.174199, time_zone_center_longitude=-120):
    """将日期时间转换为年积日+日秒"""
    pst_center_longitude = time_zone_center_longitude
    loc_longitude = longitude
    correction = np.abs(60 / 15 * (loc_longitude - pst_center_longitude))
    min_correction = int(correction)
    sec_correction = int((correction - min_correction) * 60)
    if date_and_time.minute <= min_correction:
        date_and_time = date_and_time.replace(
            hour=date_and_time.hour - 1,
            minute=60 + date_and_time.minute - min_correction - 1,
            second=60 - sec_correction
        )
    else:
        date_and_time = date_and_time.replace(
            minute=date_and_time.minute - min_correction - 1,
            second=60 - sec_correction
        )
    time_of_day = date_and_time.hour * 3600 + date_and_time.minute * 60 + date_and_time.second
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (date_and_time.year % 4 == 0) and (date_and_time.year % 100 != 0 or date_and_time.year % 400 == 0):
        months[1] = 29
    day_of_year = sum(months[:date_and_time.month - 1]) + date_and_time.day
    return day_of_year, time_of_day


def solar_angle(times, latitude=37.424107, longitude=-122.174199, time_zone_center_longitude=-120):
    """计算太阳方位角和天顶角"""
    day_of_year, time_of_day = doy_tod_conv(times, longitude, time_zone_center_longitude)
    latitude = radians(latitude)
    B = 2 * pi * (day_of_year - 80) / 365.0
    eot_minutes = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B)
    time_of_day_corrected = time_of_day + (eot_minutes * 60)
    alpha = 2 * pi * (time_of_day_corrected - 43200) / 86400
    delta = radians(23.44 * sin(radians((360 / 365.25) * (day_of_year - 80))))
    chi = acos(sin(delta) * sin(latitude) + cos(delta) * cos(latitude) * cos(alpha))
    tan_xi = sin(alpha) / (sin(latitude) * cos(alpha) - cos(latitude) * tan(delta))
    if alpha > 0 and tan_xi > 0:
        xi = pi + atan(tan_xi)
    elif alpha > 0 and tan_xi < 0:
        xi = 2 * pi + atan(tan_xi)
    elif alpha < 0 and tan_xi > 0:
        xi = atan(tan_xi)
    else:
        xi = pi + atan(tan_xi)
    return degrees(xi), degrees(chi)


def get_theo_sun_position(time_obj):
    """理论太阳位置（无畸变修正）"""
    delta, r, origin_y, origin_x = 14.036, 928, 928, 960
    azimuth, zenith = solar_angle(time_obj)
    rho = zenith / 90 * r
    theta = azimuth - delta + 90
    theo_y = origin_y - rho * sin(radians(theta))
    theo_x = origin_x + rho * cos(radians(theta))
    return int(round(theo_x)), int(round(theo_y))


# ==========================================
# 构建月度镜头畸变标定台账
# ==========================================
MONTHLY_INTERP_TABLE = {}
print("⏳ 正在使用极其纯净的晴天基准日，进行镜头畸变物理标定...")

for year, month, day in sunny_calibration_days:
    month_int = month
    date_str = f"{year}{month:02d}{day:02d}"
    tar_name = f"2019_{month:02d}_images_raw.tar"
    tar_path = os.path.join(tar_folder, tar_name)

    if not os.path.exists(tar_path):
        print(f"⚠️ 跳过 {date_str}，找不到对应的 tar 包")
        continue

    time_floats, dx_list, dy_list = [], [], []
    with tarfile.open(tar_path, "r") as tar:
        day_members = [m for m in tar.getmembers() if m.name.endswith('.jpg') and date_str in m.name]
        day_members.sort(key=lambda x: os.path.basename(x.name))

        for m in day_members[::15]:
            curr_time = dt.datetime.strptime(os.path.basename(m.name).split('.')[0], '%Y%m%d%H%M%S')
            if curr_time.hour < 7 or curr_time.hour > 18:
                continue

            img_raw = cv2.imdecode(
                np.asarray(bytearray(tar.extractfile(m).read()), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            theo_x, theo_y = get_theo_sun_position(curr_time)

            search_radius = 120
            if (theo_y - search_radius < 0 or theo_y + search_radius > 2048 or
                    theo_x - search_radius < 0 or theo_x + search_radius > 2048):
                continue

            roi = gray[theo_y - search_radius: theo_y + search_radius,
                       theo_x - search_radius: theo_x + search_radius]
            _, thresh = cv2.threshold(roi, 245, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                cv2.erode(thresh, np.ones((7, 7), np.uint8), iterations=2),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 50:
                    (cX_local, cY_local), _ = cv2.minEnclosingCircle(largest_contour)
                    time_float = curr_time.hour + curr_time.minute / 60.0 + curr_time.second / 3600.0
                    time_floats.append(round(time_float, 3))
                    dx_list.append(round((theo_x - search_radius + cX_local) - theo_x, 1))
                    dy_list.append(round((theo_y - search_radius + cY_local) - theo_y, 1))

    if time_floats:
        MONTHLY_INTERP_TABLE[month_int] = {'times': time_floats, 'dx': dx_list, 'dy': dy_list}

# 回退机制
FALLBACK_MONTHS = {2: 1, 3: 1, 4: 1, 11: 10, 12: 10}


def get_smart_sun_position(time_obj):
    """获取经镜头畸变修正后的太阳位置"""
    theo_x, theo_y = get_theo_sun_position(time_obj)
    month = time_obj.month
    ref_month = month if month in MONTHLY_INTERP_TABLE else FALLBACK_MONTHS.get(month, 1)

    if ref_month not in MONTHLY_INTERP_TABLE:
        return theo_x, theo_y

    table = MONTHLY_INTERP_TABLE[ref_month]
    t_curr = time_obj.hour + time_obj.minute / 60.0 + time_obj.second / 3600.0
    dx = np.interp(t_curr, table['times'], table['dx'])
    dy = np.interp(t_curr, table['times'], table['dy'])
    return int(round(theo_x + dx)), int(round(theo_y + dy))


print("✅ 镜头畸变标定完成！")

# ==========================================
# 🌟 图像双分支采样核心 (与 V4 完全一致)
# ==========================================
def extract_dual_branch_images(img_bgr, sun_x, sun_y):
    """提取双分支图像 (严格 PyTorch C,H,W 格式)"""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (1053, 1024), 992, 255, -1)
    img_clean = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    # 1. 全局图 (缩放并转换通道)
    global_img_rgb = cv2.cvtColor(
        cv2.resize(img_clean, (256, 256), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB
    )
    img_g_pt = global_img_rgb.transpose(2, 0, 1)  # -> (3, 256, 256)

    # 2. 局部图 (极宽容裁剪法)
    HUGE_CROP = 768
    HALF_CROP = HUGE_CROP // 2
    canvas = np.zeros((HUGE_CROP, HUGE_CROP, 3), dtype=np.uint8)

    y_min, y_max = sun_y - HALF_CROP, sun_y + HALF_CROP
    x_min, x_max = sun_x - HALF_CROP, sun_x + HALF_CROP
    img_h, img_w = img_clean.shape[:2]

    img_y_min, img_y_max = max(0, y_min), min(img_h, y_max)
    img_x_min, img_x_max = max(0, x_min), min(img_w, x_max)

    if img_y_min < img_y_max and img_x_min < img_x_max:
        canvas_y_min, canvas_x_min = img_y_min - y_min, img_x_min - x_min
        canvas[canvas_y_min:canvas_y_min + (img_y_max - img_y_min),
               canvas_x_min:canvas_x_min + (img_x_max - img_x_min)] = \
            img_clean[img_y_min:img_y_max, img_x_min:img_x_max]

    local_img_rgb = cv2.cvtColor(
        cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB
    )
    img_l_pt = local_img_rgb.transpose(2, 0, 1)  # -> (3, 128, 128)

    return img_g_pt, img_l_pt


# ==========================================
# 工具函数
# ==========================================
def find_time_within_pdseries(time_array, time_point):
    """在 pd.Series 索引中查找精确时间点"""
    probable_idx = np.searchsorted(time_array, time_point)
    if probable_idx < len(time_array) and time_array[probable_idx] == time_point:
        return probable_idx
    return None


# ==========================================
# 🔴 V5 核心改造 A: 写入临时 H5（所有帧先写入）
# ==========================================
print("\n" + "=" * 60)
print("🚀 V5 图像数据集生成引擎启动")
print("=" * 60)

pv_output_all = pd.read_pickle(pv_data_path)
print(f"📊 加载 PV 数据: {len(pv_output_all)} 条记录")

# 内存缓冲机制
BUFFER_SIZE = 100
buffers = {'global': [], 'local': [], 'pv': [], 'sun': [], 'times': []}
all_recorded_times = []


def flush_buffer(grp):
    """将缓冲区的数据刷入 HDF5"""
    if not buffers['pv']:
        return
    n_new = len(buffers['pv'])
    curr_len = grp['pv_log'].shape[0]

    for key in ['global_images_log', 'local_images_log', 'pv_log', 'sun_pos', 'times_log']:
        grp[key].resize(curr_len + n_new, axis=0)

    grp['global_images_log'][curr_len:] = np.stack(buffers['global'])
    grp['local_images_log'][curr_len:] = np.stack(buffers['local'])
    grp['pv_log'][curr_len:] = np.array(buffers['pv'], dtype=np.float32)
    grp['sun_pos'][curr_len:] = np.array(buffers['sun'], dtype=np.int32)
    grp['times_log'][curr_len:] = np.array(buffers['times'], dtype=np.float64)

    for k in buffers.keys():
        buffers[k].clear()


# ------------------------------------------------
# 第一步: 遍历所有 tar 包，写入临时 H5
# ------------------------------------------------
print(f"\n📝 第一步: 写入临时 H5 文件...")

with h5py.File(temp_h5_path, 'w') as h5f_temp:
    # 临时文件也使用 trainval 组（仅用于汇聚）
    grp_temp = h5f_temp.create_group('trainval')
    grp_temp.create_dataset(
        'global_images_log', shape=(0, 3, 256, 256),
        maxshape=(None, 3, 256, 256), dtype='uint8',
        chunks=(BUFFER_SIZE, 3, 256, 256), compression='lzf'
    )
    grp_temp.create_dataset(
        'local_images_log', shape=(0, 3, 128, 128),
        maxshape=(None, 3, 128, 128), dtype='uint8',
        chunks=(BUFFER_SIZE, 3, 128, 128), compression='lzf'
    )
    grp_temp.create_dataset(
        'pv_log', shape=(0,), maxshape=(None,),
        dtype='float32', chunks=(BUFFER_SIZE,)
    )
    grp_temp.create_dataset(
        'sun_pos', shape=(0, 2), maxshape=(None, 2),
        dtype='int32', chunks=(BUFFER_SIZE, 2)
    )
    # 🔴 V5 新增: 时间戳对齐字段 (Unix 秒, float64)
    grp_temp.create_dataset(
        'times_log', shape=(0,), maxshape=(None,),
        dtype='float64', chunks=(BUFFER_SIZE,)
    )

    tar_files = sorted([f for f in os.listdir(tar_folder) if f.endswith('.tar')])

    for tar_name in tar_files:
        print(f"\n--- ⚡ 正在极速解析: {tar_name} ---")
        tar_path = os.path.join(tar_folder, tar_name)
        with tarfile.open(tar_path, "r") as tar:
            members = sorted(
                [m for m in tar.getmembers() if m.name.endswith('.jpg')],
                key=lambda x: x.name
            )
            prev_img_gray, current_day = None, None

            for m in tqdm(members, desc="提取与写入"):
                try:
                    curr_time = dt.datetime.strptime(
                        os.path.basename(m.name).split('.')[0], '%Y%m%d%H%M%S'
                    )
                except:
                    continue

                if curr_time.date() != current_day:
                    current_day, prev_img_gray = curr_time.date(), None

                # 🔴 V5 改造 A: 废除 hour 硬编码，改为 PV 功率阈值过滤
                pv_idx = find_time_within_pdseries(pv_output_all.index, curr_time)
                if pv_idx is None:
                    continue

                # -------- V4 此处为 hour 硬编码过滤 (已删除) --------
                # if curr_time.hour < 7 or (curr_time.hour == 7 and curr_time.minute < 30) or \
                #    curr_time.hour > 17 or (curr_time.hour == 17 and curr_time.minute > 30): continue
                # ----------------------------------------------------

                # 🔴 V5 新逻辑: 基于 PV 功率的物理阈值过滤
                try:
                    pv_val = float(pv_output_all.iloc[pv_idx])
                except (ValueError, IndexError):
                    continue  # 处理 NaN 或索引异常

                if pv_val < 0.05:
                    continue  # PV 功率过低，跳过该帧

                # 内存解压
                img_raw = cv2.imdecode(
                    np.asarray(bytearray(tar.extractfile(m).read()), dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                # 卡死帧过滤
                curr_img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                if prev_img_gray is not None and \
                   np.sum(np.abs(curr_img_gray.astype(np.int16) - prev_img_gray.astype(np.int16))) == 0:
                    continue
                prev_img_gray = curr_img_gray

                # 计算太阳位置与双分支图像
                sun_x, sun_y = get_smart_sun_position(curr_time)
                img_g_pt, img_l_pt = extract_dual_branch_images(img_raw, sun_x, sun_y)

                # 填入缓冲区
                buffers['global'].append(img_g_pt)
                buffers['local'].append(img_l_pt)
                buffers['pv'].append(pv_val)
                buffers['sun'].append([sun_x, sun_y])
                # 🔴 V5: 将 datetime 转为 Unix 秒 (float64) 写入 times_log
                epoch = dt.datetime(1970, 1, 1)
                buffers['times'].append((curr_time - epoch).total_seconds())
                all_recorded_times.append(curr_time)

                if len(buffers['pv']) >= BUFFER_SIZE:
                    flush_buffer(grp_temp)

    # 刷入最后残留数据
    flush_buffer(grp_temp)

# ==========================================
# 🔴 V5 核心改造 B: 8:2 物理隔离切分
# ==========================================
print(f"\n{'=' * 60}")
print("🔄 第二步: 执行 8:2 物理隔离切分...")
print(f"{'=' * 60}")

# 🔴 强制硬编码 Chunk 配置，杜绝 HDF5 默认行为！
V5_CHUNK_MAP = {
    'global_images_log': (100, 3, 256, 256),
    'local_images_log': (100, 3, 128, 128),
    'pv_log': (100,),
    'sun_pos': (100, 2),
    'times_log': (100,)  # 🔴 V5 新增: 时间戳对齐
}


def repackage_to_v5(temp_path, final_path, split_idx):
    """
    🔴 V5 内聚重构函数:
      1. 读取临时 H5
      2. 按时间顺序 8:2 切分为 trainval / test
      3. 强制 I/O 参数: compression='lzf', chunks=100, shuffle=True(图像)
      4. 完成后删除临时文件
    """
    print(f"\n🔄 正在执行 V5 物理隔离重构...")
    print(f"   临时文件: {os.path.basename(temp_path)}")
    print(f"   最终文件: {os.path.basename(final_path)}")
    print(f"   切分点: trainval={split_idx} 帧, test=剩余帧")

    with h5py.File(temp_path, 'r') as src, h5py.File(final_path, 'w') as dst:
        grp_src = src['trainval']

        # 创建目标双组
        grp_train = dst.create_group('trainval')
        grp_test = dst.create_group('test')

        for key in grp_src.keys():
            dataset = grp_src[key]
            total_len = dataset.shape[0]
            my_chunk = V5_CHUNK_MAP.get(key, True)

            # 图像数据集开启 shuffle
            enable_shuffle = 'images' in key

            print(f"  ✂️ 切分 [{key}]  (总帧数: {total_len})...")

            # --- 写入 trainval ---
            grp_train.create_dataset(
                key,
                data=dataset[:split_idx],
                compression='lzf',
                shuffle=enable_shuffle,
                chunks=my_chunk
            )

            # --- 写入 test ---
            grp_test.create_dataset(
                key,
                data=dataset[split_idx:],
                compression='lzf',
                shuffle=enable_shuffle,
                chunks=my_chunk
            )

    # 验证: 检查总帧数是否一致
    with h5py.File(final_path, 'r') as f_check:
        train_count = f_check['trainval']['pv_log'].shape[0]
        test_count = f_check['test']['pv_log'].shape[0]
        total_check = train_count + test_count

    # 🗑️ 删除临时文件
    os.remove(temp_path)
    print(f"   🗑️ 临时文件已删除")

    print(f"\n✅ V5 物理隔离重构完成!")
    print(f"   trainval: {train_count} 帧 | test: {test_count} 帧 | 总计: {total_check} 帧")
    return train_count, test_count


# 计算 8:2 切分点
total_frames = len(all_recorded_times)
split_idx = int(total_frames * 0.8)

print(f"\n📊 帧统计:")
print(f"   总帧数: {total_frames}")
print(f"   8:2 切分点: {split_idx} / {total_frames - split_idx}")

# 执行重构
train_count, test_count = repackage_to_v5(temp_h5_path, final_h5_path, split_idx)

# 🔴 V5: 时间戳已嵌入 H5 内部 times_log，不再需要外部 .npy 文件

print(f"\n{'=' * 60}")
print(f"🎉 V5 图像数据集生成完毕!")
print(f"📁 最终文件: {final_h5_path}")
print(f"   trainval: {train_count} 帧 | test: {test_count} 帧")
print(f"📁 时间戳已内嵌于 H5: trainval/times_log, test/times_log")
print(f"{'=' * 60}")
