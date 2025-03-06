# 姿势追踪工具

这是一个基于 YOLO 和 MediaPipe 的人体姿势检测与 3D 轨迹追踪项目，包含两个 Python 脚本，分别用于视频文件处理和摄像头实时检测。

## 脚本说明

- **`VideoPoseTracker.py`（视频姿势追踪器）**  
  用于从预录制的视频文件（如 `7.mp4`）中检测人体姿势，并生成 3D 运动轨迹，结果保存为 `output.mp4`。

- **`RealTimePoseTracker.py`（实时姿势追踪器）**  
  使用摄像头进行实时人体姿势检测，并生成 3D 运动轨迹，结果保存为 `output_webcam.mp4`。

## 依赖安装

运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt# PoseTracking
