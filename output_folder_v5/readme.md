===========================
  2019_dataset_dual_V5.h5  
===========================

┌ trainval: 样本数 100,167
│  global_images_log          (100167, 3, 256, 256)           uint8     chunks=(100, 3, 256, 256)  lzf+shuffle=yes
│  local_images_log           (100167, 3, 128, 128)           uint8     chunks=(100, 3, 128, 128)  lzf+shuffle=yes
│  pv_log                     (100167)                        float32   chunks=(100)  lzf+shuffle=no
│  sun_pos                    (100167, 2)                     int32     chunks=(100, 2)  lzf+shuffle=no
│  times_log                  (100167)                        float64   chunks=(100)  lzf+shuffle=no
│
│  📊 帧间隔统计 (秒):
│     范围:  10.0 ~ 8602550.0
│     均值:  204.7   中位数: 60.0
│     标准差: 27251.6
│     断层 (>120s): 153 处
│     时间跨度: 2019-01-24 15:32 → 2019-09-18 21:51
│     跨越天数: 237.3 天

┌ test: 样本数 25,042
│  global_images_log          (25042, 3, 256, 256)            uint8     chunks=(100, 3, 256, 256)  lzf+shuffle=yes
│  local_images_log           (25042, 3, 128, 128)            uint8     chunks=(100, 3, 128, 128)  lzf+shuffle=yes
│  pv_log                     (25042)                         float32   chunks=(100)  lzf+shuffle=no
│  sun_pos                    (25042, 2)                      int32     chunks=(100, 2)  lzf+shuffle=no
│  times_log                  (25042)                         float64   chunks=(100)  lzf+shuffle=no
│
│  📊 帧间隔统计 (秒):
│     范围:  10.0 ~ 50300.0
│     均值:  131.7   中位数: 60.0
│     标准差: 1841.9
│     断层 (>120s): 42 处
│     时间跨度: 2019-09-18 21:52 → 2019-10-27 01:57
│     跨越天数: 38.2 天

===========================
  2019_dataset_flow_V5.h5  
===========================
┌ trainval  global_flow_log            (100167, 2, 256, 256)           float16   chunks=(100, 2, 256, 256)  lzf+shuffle=yes
┌ test      global_flow_log            (25042, 2, 256, 256)            float16   chunks=(100, 2, 256, 256)  lzf+shuffle=yes

========
  交叉校验  
========
✅ trainval: 图像 100,167 帧  ==  光流 100,167 帧  ✓
✅ test: 图像 25,042 帧  ==  光流 25,042 帧  ✓

图像 H5 分割: trainval 80.0%  vs  test 20.0%

🎉 全部校验通过！两个 H5 文件完全一致。

======================================================================
  检查完毕！
======================================================================