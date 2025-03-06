# 姿势追踪工具

这是一个人体姿势检测与 3D 轨迹追踪项目，包含4个 Python 脚本，分别用于视频文件处理和摄像头实时检测。

## YOLO + MediaPipe 与 YOLOv8-pose 的对比

本项目提供了两种姿势检测实现方式，以下是对比分析：

| **特性**               | **YOLO + MediaPipe**                          | **YOLOv8-pose**                          |
|-----------------------|----------------------------------------------|-----------------------------------------|
| **流程**              | YOLO 检测目标框，MediaPipe 提取 33 个关键点    | 单模型一步完成目标检测和 17 个关键点提取 |
| **关键点数量**         | 33 个（含 Z 轴深度）                          | 17 个（COCO 格式，无 Z 轴）             |
| **依赖**              | Ultralytics + MediaPipe                      | 仅 Ultralytics                         |
| **计算效率**           | 较高开销（两步处理）                          | 较低开销（一步处理）                    |
| **实时性**            | 稍逊（受双重处理影响）                        | 更优（单模型效率高）                    |
| **3D 轨迹支持**        | 原生支持（MediaPipe 提供 Z 轴）               | 需额外深度估计（Z 轴暂为 0）            |
| **适用场景**           | 高精度姿势分析、需要深度信息                  | 实时监控、基本姿势跟踪                  |

### 选择建议
- 若需要详细的关键点信息和 3D 深度（如运动分析），推荐使用 `VideoPoseTracker.py` 或 `RealTimePoseTracker.py`。
- 若追求实时性和部署简便性（如监控系统），推荐使用 `VideoYoloPoseTracker.py` 或 `RealTimeYoloPoseTracker.py`。

## 脚本说明
- **`VideoPoseTracker.py`**：基于 YOLO 和 MediaPipe 的视频姿势检测，输出 `output.mp4`。
- **`RealTimePoseTracker.py`**：基于 YOLO 和 MediaPipe 的摄像头实时检测，输出 `output_webcam.mp4`。
- **`VideoYoloPoseTracker.py`**：使用 YOLOv8-pose 的视频姿势检测，输出 `output.mp4`。
- **`RealTimeYoloPoseTracker.py`**：使用 YOLOv8-pose 的摄像头实时检测，输出 `output_webcam.mp4`。

- **`VideoPoseTracker.py`（视频姿势追踪器）**  
  用于从预录制的视频文件（如 `7.mp4`）中检测人体姿势，并生成 3D 运动轨迹，结果保存为 `output.mp4`。

- **`RealTimePoseTracker.py`（实时姿势追踪器）**  
  使用摄像头进行实时人体姿势检测，并生成 3D 运动轨迹，结果保存为 `output_webcam.mp4`。

## 依赖安装

运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt# PoseTracking
