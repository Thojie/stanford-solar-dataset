import os
import subprocess
import time

project_path = os.getcwd()
script_folder = os.path.join(project_path, 'v5_dataprocess')

def run_task(command, task_name):
    print(f"\n{'='*50}")
    print(f"⏳ 正在启动: {task_name}...")
    print(f"{'='*50}\n")
    
    # 启动子进程执行命令
    result = subprocess.run(command, shell=True)
    
    # 检查是否报错崩溃
    if result.returncode != 0:
        print(f"\n🚨 警告：{task_name} 执行失败！流水线已紧急中止。")
        exit(1)
        
    print(f"\n✅ {task_name} 圆满完成！\n")
    time.sleep(3) # 稍微喘口气，让内存和硬盘释放一下缓存

if __name__ == '__main__':
    print("🚀 睡梦级全自动流水线 - V5 版本")
    print("=" * 50)
    
    # ==========================================
    # V5 管线执行顺序:
    #   1. PV 数据清洗 → V5_pv_valid.pkl
    #   2. 图像数据集生成 → 2019_dataset_dual_V5.h5
    #   3. 光流计算 → 2019_dataset_flow_V5.h5
    # ==========================================
    
    # 任务 1: PV 数据清洗
    run_task(
        f"python \"{os.path.join(script_folder, 'v5_data_process_pv.py')}\"",
        "阶段 1: PV 数据清洗 (V5_pv_valid.pkl)"
    )
    
    # 任务 2: 图像数据集生成
    run_task(
        f"python \"{os.path.join(script_folder, 'v5_data_process_image.py')}\"",
        "阶段 2: 图像数据集生成 (2019_dataset_dual_V5.h5)"
    )
    
    # 任务 3: 光流计算
    run_task(
        f"python \"{os.path.join(script_folder, 'v5_data_process_flow.py')}\"",
        "阶段 3: 光流计算 (2019_dataset_flow_V5.h5)"
    )

    print("\n🎉 V5 全部数据管线已在无人值守状态下通关！")
    
    # 🌟 终极护肝大招：跑完自动关机！
    # 如果你想跑完让电脑自动关机省电，把下面这行代码的注释去掉
    # os.system("shutdown /s /t 60") # 60秒后自动关机
