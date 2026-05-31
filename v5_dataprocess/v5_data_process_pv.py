"""
V5 PV Data Processing Script
=======================================
优化目标：
  - 维持 V4 的插值与过滤逻辑
  - 夜间过滤阈值从 PV > 0 改为 PV > 0.05（50W）
  - 全天候保留，无 hour 硬编码
  - 输出 V5_pv_valid.pkl

输出: pv_data/V5_pv_valid.pkl
"""

import numpy as np
import datetime
import pandas as pd
import os
from scipy.interpolate import interp1d

# ==========================================
# 配置
# ==========================================
project_path = os.getcwd()
input_folder = os.path.join(project_path, 'pv_data')
output_folder = os.path.join(project_path, 'pv_data')
os.makedirs(output_folder, exist_ok=True)

pv_data_raw_path = os.path.join(input_folder, '2019_pv_raw.csv')

start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2019, 12, 31)

print("=" * 60)
print("V5 PV Data Processing")
print("=" * 60)

# ==========================================
# 1. 读取原始 CSV
# ==========================================
print("\n[1/5] 读取原始 PV 数据...")
pv_data_raw_df = pd.read_csv(pv_data_raw_path)

pv_data_df = pd.DataFrame()
pv_data_df["Time"] = pv_data_raw_df["Date"].apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")
)
pv_data_df["PV_output_kw"] = pv_data_raw_df["Huang_E4102_kW"]

# 筛选 2019 年范围
pv_data_df = pv_data_df[
    (pv_data_df['Time'] >= start_date) & (pv_data_df['Time'] <= end_date)
]
pv_data_df.reset_index(drop=True, inplace=True)
pv_data_df = pv_data_df.set_index('Time')
print(f"   原始记录数: {len(pv_data_df)}")

# ==========================================
# 2. 插值到 1秒 → 重采样到 10秒
# ==========================================
print("\n[2/5] 执行 10 秒插值重采样...")

def interpaverage(record_times, record_outputs):
    """插值到 1秒后 10秒平均"""
    start_time_dt = record_times[0]
    start_time_str = datetime.datetime.strftime(start_time_dt, "%Y-%m-%dT%H:%M:%S")
    record_times_dt64 = record_times.astype('datetime64[s]')
    record_times_elapsed = (record_times_dt64 - np.datetime64(start_time_str)).astype('int')

    f = interp1d(record_times_elapsed, record_outputs)

    int_times_elapsed = np.arange(0, record_times_elapsed[-1], 1)
    int_outputs = f(int_times_elapsed)

    int_datetimes = start_time_dt + int_times_elapsed * datetime.timedelta(seconds=1)
    pv_output_raw_int = pd.Series(int_outputs, index=int_datetimes)

    pv_output_10s_int = pv_output_raw_int.resample('10s').mean()
    assert np.sum((pv_output_10s_int.index.second % 10).values) == 0
    return pv_output_10s_int

record_times = np.asarray([time.to_pydatetime() for time in pv_data_df.index])
record_outputs = pv_data_df['PV_output_kw'].values

print(f"   record_times.shape:   {record_times.shape}")
print(f"   record_outputs.shape: {record_outputs.shape}")

pv_output_10s_int = interpaverage(record_times, record_outputs)
print(f"   插值后 10s 记录数: {len(pv_output_10s_int)}")

# ==========================================
# 3. 手动标记的坏点过滤
# ==========================================
print("\n[3/5] 标记已知坏点...")
invalid_dates = {
    'start': [
        datetime.datetime(2019, 1, 28, 10, 29, 0),
        datetime.datetime(2019, 5, 22, 14, 18, 0),
        datetime.datetime(2019, 10, 27)
    ],
    'end': [
        datetime.datetime(2019, 1, 28, 11, 47, 0),
        datetime.datetime(2019, 5, 22, 17, 3, 0),
        datetime.datetime(2019, 10, 28)
    ]
}

invalid_dates_mask = pd.Series(False, index=pv_output_10s_int.index)
for i in range(len(invalid_dates["start"])):
    start_idx = pd.Timestamp(invalid_dates['start'][i])
    end_idx = pd.Timestamp(invalid_dates['end'][i])
    invalid_dates_mask.loc[start_idx:end_idx] = True

# ==========================================
# 4. 缺失记录过滤 (插值区间 > 1 小时)
# ==========================================
print("\n[4/5] 过滤缺失记录(>1h 间隔)...")
record_interval = record_times[1:] - record_times[:-1]
record_invalid_start = record_times[:-1][record_interval > np.timedelta64(1, 'h')]
record_invalid_end = record_times[1:][record_interval > np.timedelta64(1, 'h')]

missing_mask = pd.Series(False, index=pv_output_10s_int.index)
for start, end in zip(record_invalid_start, record_invalid_end):
    missing_mask.loc[start:end] = True

# ==========================================
# 5. 🔴 V5 改造: 夜间过滤改为 PV > 0.05
# ==========================================
print("\n[5/5] 应用 V5 物理阈值过滤 (PV_output > 0.05)...")
# V4: pv_output_10s_int > 0
# V5: pv_output_10s_int > 0.05  (50W threshold)
pv_output_valid = pv_output_10s_int[
    (~invalid_dates_mask) & (~missing_mask) & (pv_output_10s_int > 0.05)
]

print(f"\n{'=' * 60}")
print(f"📊 V5 有效 PV 记录数: {len(pv_output_valid)}")
print(f"   相比于原始 10s 数据保留: {len(pv_output_valid) / len(pv_output_10s_int) * 100:.1f}%")
print(f"{'=' * 60}")

# 保存
output_path = os.path.join(output_folder, 'V5_pv_valid.pkl')
pv_output_valid.to_pickle(output_path)
print(f"\n✅ V5 PV 数据已保存至: {output_path}")
print("🎉 V5 PV 数据清洗完成！")
