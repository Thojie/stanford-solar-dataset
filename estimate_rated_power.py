import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 读取数据
df = pd.read_csv('pv_data/2019_pv_raw.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 提取功率数据
power = df['Huang_E4102_kW'].values

# 基本统计
print("="*60)
print("光伏电站功率数据分析")
print("="*60)
print(f"数据点总数: {len(df)}")
print(f"时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
print(f"\n功率数据统计 (单位: kW):")
print(f"  最小值: {power.min():.4f}")
print(f"  最大值: {power.max():.4f}")
print(f"  平均值: {power.mean():.4f}")
print(f"  中位数: {np.median(power):.4f}")
print(f"  标准差: {power.std():.4f}")

# 提取白天时间（假设白天是6:00-18:00）
df['hour'] = df['Date'].dt.hour
daytime = df[(df['hour'] >= 6) & (df['hour'] < 18)]
print(f"\n白天数据（6:00-18:00）:")
print(f"  数据点数: {len(daytime)}")
print(f"  最大功率: {daytime['Huang_E4102_kW'].max():.4f} kW")
print(f"  平均功率: {daytime['Huang_E4102_kW'].mean():.4f} kW")

# 筛选正功率数据（排除夜间噪声）
positive_power = df[df['Huang_E4102_kW'] > 0]
print(f"\n正功率数据（功率 > 0）:")
print(f"  数据点数: {len(positive_power)}")
if len(positive_power) > 0:
    print(f"  最大值: {positive_power['Huang_E4102_kW'].max():.4f} kW")
    print(f"  平均值: {positive_power['Huang_E4102_kW'].mean():.4f} kW")
    print(f"  95分位数: {positive_power['Huang_E4102_kW'].quantile(0.95):.4f} kW")
    print(f"  99分位数: {positive_power['Huang_E4102_kW'].quantile(0.99):.4f} kW")

# 估计额定功率
print("\n"+"="*60)
print("额定功率估计:")
print("="*60)
max_power = power.max()
print(f"方法1 - 全数据集最大值: {max_power:.4f} kW ≈ {round(max_power)} kW")

if len(positive_power) > 0:
    max_positive = positive_power['Huang_E4102_kW'].max()
    print(f"方法2 - 正功率最大值: {max_positive:.4f} kW ≈ {round(max_positive)} kW")
    
    q99 = positive_power['Huang_E4102_kW'].quantile(0.99)
    print(f"方法3 - 正功率99分位: {q99:.4f} kW ≈ {round(q99)} kW")

# 白天最大功率
daytime_max = daytime['Huang_E4102_kW'].max()
print(f"方法4 - 白天(6-18时)最大值: {daytime_max:.4f} kW ≈ {round(daytime_max)} kW")

# 绘制功率分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 全时段功率时间序列（采样）
axes[0, 0].plot(df['Date'][::60], power[::60], alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('功率时间序列（每小时采样）')
axes[0, 0].set_ylabel('功率 (kW)')
axes[0, 0].grid(True, alpha=0.3)

# 按小时的平均功率
hourly_avg = df.groupby('hour')['Huang_E4102_kW'].mean()
axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color='steelblue')
axes[0, 1].set_title('按小时的平均功率')
axes[0, 1].set_xlabel('小时')
axes[0, 1].set_ylabel('平均功率 (kW)')
axes[0, 1].grid(True, alpha=0.3)

# 功率直方图
axes[1, 0].hist(power, bins=100, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('功率分布直方图')
axes[1, 0].set_xlabel('功率 (kW)')
axes[1, 0].set_ylabel('频数')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 正功率分布
if len(positive_power) > 0:
    axes[1, 1].hist(positive_power['Huang_E4102_kW'], bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(positive_power['Huang_E4102_kW'].max(), color='red', linestyle='--', 
                       linewidth=2, label=f'最大值: {positive_power["Huang_E4102_kW"].max():.2f}kW')
    axes[1, 1].set_title('正功率分布直方图')
    axes[1, 1].set_xlabel('功率 (kW)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pv_data/power_analysis.png', dpi=100, bbox_inches='tight')
print("\n图表已保存到: pv_data/power_analysis.png")

# 月度最大功率趋势
df['date'] = df['Date'].dt.date
df['month'] = df['Date'].dt.to_period('M')
monthly_max = df.groupby('month')['Huang_E4102_kW'].max()
print(f"\n月度最大功率（逐月）:")
print(monthly_max)
