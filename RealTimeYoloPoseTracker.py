import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import threading
import queue

# 强制使用 TkAgg 后端
matplotlib.use('TkAgg')

# 删除所有MediaPipe相关导入和初始化

def generate_color(track_id):
    """生成基于track_id的固定颜色"""
    golden_ratio = 0.618033988749895
    hue = (hash(track_id) * golden_ratio) % 1.0
    return plt.cm.hsv([hue])[0][:3]

class VideoProcessor:
    def __init__(self):
        self.detector = None
        self.cap = None
        self.trajectories = defaultdict(list)
        self.kalman_filters = dict()
        self.frame_size = None
        self.track_lifespan = defaultdict(int)
        self.out_video = None
        self.last_processed_frame = None

        # YOLOv8-pose关键点定义（COCO格式）
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16)
        ]

    def create_kf(self):
        """初始化卡尔曼滤波器（保持不变）"""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]])
        kf.H = np.eye(3,6)
        kf.P *= 1000
        kf.R = np.eye(3) * 10
        kf.Q = np.eye(6) * 0.01
        return kf

    def process_frame(self, frame):
        """使用YOLOv8-pose进行处理的修改版本"""
        start_time = time.time()
        results = self.detector.track(frame, persist=True, iou=0.7, verbose=False)
        print(f"Detection time: {time.time() - start_time:.2f} seconds")
        
        output_data = {"frame": None, "trajectories": {}}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()

            for box, tid, kpts in zip(boxes, track_ids, keypoints):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制关键点和连接线
                for i, (x, y) in enumerate(kpts):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                for conn in self.connections:
                    start = tuple(map(int, kpts[conn[0]]))
                    end = tuple(map(int, kpts[conn[1]]))
                    if start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
                        cv2.line(frame, start, end, (255, 0, 0), 2)

                # 计算髋部中心（关键点11和12）
                left_hip = kpts[11]
                right_hip = kpts[12]
                if left_hip[0] > 0 and right_hip[0] > 0:
                    hip_center = ((left_hip + right_hip) / 2).astype(int)
                    self.update_trajectory(tid, hip_center)
                    
                    # 绘制髋部中心
                    cv2.circle(frame, tuple(hip_center), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Hip Center", 
                               (hip_center[0] + 10, hip_center[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 轨迹生命周期管理（保持不变）
            for tid in list(self.trajectories.keys()):
                if tid not in track_ids:
                    self.track_lifespan[tid] += 1
                    if self.track_lifespan[tid] > 10000:
                        del self.trajectories[tid]
                        del self.kalman_filters[tid]
                else:
                    self.track_lifespan[tid] = 0

        output_data["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_data["trajectories"] = dict(self.trajectories)
        self.last_processed_frame = frame
        return output_data

    def update_trajectory(self, tid, position):
        """更新3D轨迹（简化Z轴处理）"""
        x, y = position
        z = 0  # 这里可以添加深度估计算法
        
        if tid not in self.trajectories:
            self.trajectories[tid] = []
        self.trajectories[tid].append([x, y, z])
        
        # 卡尔曼滤波（保持不变）
        if tid not in self.kalman_filters:
            self.kalman_filters[tid] = self.create_kf()
            self.kalman_filters[tid].x = np.array([x, y, z, 0, 0, 0])
        
        kf = self.kalman_filters[tid]
        kf.predict()
        kf.update([x, y, z])
        smoothed_pos = kf.x[:3].tolist()
        
        # 保持轨迹长度
        self.trajectories[tid][-1] = smoothed_pos
        if len(self.trajectories[tid]) > 50:
            self.trajectories[tid].pop(0)

    def initialize(self):
        """初始化资源（修改模型路径）"""
        model_path = 'yolov8m-pose.pt'  # 改为pose模型
        if os.path.exists(model_path):
            print(f"Loading local model from {model_path}")
            self.detector = YOLO(model_path)
        else:
            print("Local model not found, downloading yolov8m-pose.pt")
            self.detector = YOLO('yolov8m-pose.pt')

        # 初始化摄像头（保持不变）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = 30
        print(f"Webcam loaded: {self.frame_size}, {self.fps:.2f} FPS")

        # 初始化视频输出（保持不变）
        output_path = "output_webcam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        return True

    def run(self, visualizer):
        """多线程运行，主线程负责可视化和保存结果"""
        if not self.initialize():
            return

        frame_queue = queue.Queue(maxsize=30)
        processed_queue = queue.Queue(maxsize=30)
        stop_event = threading.Event()

        def read_frames():
            """从摄像头读取帧"""
            frame_count = 0
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame from webcam")
                    stop_event.set()
                    break
                frame_count += 1
                print(f"Read frame {frame_count}")
                try:
                    frame_queue.put(frame, timeout=1.0)
                except queue.Full:
                    print(f"Frame {frame_count} dropped: frame_queue full")
                time.sleep(1.0 / self.fps)  # 匹配目标帧率

        def process_frames():
            """处理视频帧"""
            frame_count = 0
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=1.0)
                    frame_count += 1
                    print(f"Processing frame {frame_count}")
                    processed = self.process_frame(frame)
                    try:
                        processed_queue.put(processed, timeout=1.0)
                    except queue.Full:
                        print(f"Processed frame {frame_count} dropped: processed_queue full")
                    frame_queue.task_done()
                except queue.Empty:
                    continue

        t1 = threading.Thread(target=read_frames, daemon=True)
        t2 = threading.Thread(target=process_frames, daemon=True)

        t1.start()
        t2.start()

        fig = visualizer.fig
        plt.show(block=False)
        frame_count = 0
        last_time = time.time()
        try:
            while True:  # 持续运行直到手动停止
                try:
                    processed = processed_queue.get(timeout=1.0)
                    frame_count += 1
                    print(f"Visualizing frame {frame_count}")
                    visualizer.update_plot(processed)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    # 保存处理后的帧到视频
                    frame_to_save = cv2.cvtColor(processed["frame"], cv2.COLOR_RGB2BGR)
                    self.out_video.write(frame_to_save)

                    current_time = time.time()
                    elapsed = current_time - last_time
                    sleep_time = max(0.033 - elapsed, 0)
                    time.sleep(sleep_time)
                    last_time = current_time
                    processed_queue.task_done()
                except queue.Empty:
                    time.sleep(0.033)  # 等待新帧
        except KeyboardInterrupt:
            print("Stopping due to user interrupt")
            stop_event.set()

        t1.join()
        t2.join()

        print("Video processing stopped. Window will close in 5 seconds.")
        time.sleep(5)
        self.cap.release()
        self.out_video.release()
        # self.pose_estimator.close()
        plt.close(fig)

class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_video = self.fig.add_subplot(121)
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.video_img = self.ax_video.imshow(np.zeros((480, 640, 3)))
        self.lines = defaultdict(lambda: self.ax_3d.plot([],[],[], lw=2, alpha=0.8)[0])
        self.color_cache = {}
        self.frame_size = None
        
        self.ax_3d.view_init(elev=30, azim=-45)
        self.ax_3d.set_xlabel('X Position')
        self.ax_3d.set_ylabel('Y Position')
        self.ax_3d.set_zlabel('Depth')
        self.ax_3d.set_box_aspect([1,1,1])
        plt.sca(self.ax_video)
        plt.axis('off')
        plt.title('Real-time Detection')
        self.ax_3d.set_title('3D Motion Trajectory')

    def update_plot(self, frame_data):
        """更新可视化内容"""
        frame = frame_data["frame"]
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]
            self.video_img = self.ax_video.imshow(np.zeros((self.frame_size[0], self.frame_size[1], 3)))
            self.ax_3d.set_xlim(0, self.frame_size[1])
            self.ax_3d.set_ylim(0, self.frame_size[0])
            self.ax_3d.set_zlim(-self.frame_size[1], self.frame_size[1])

        self.video_img.set_array(frame)
        
        for tid, traj in frame_data["trajectories"].items():
            if len(traj) < 2: continue
            
            if tid not in self.color_cache:
                self.color_cache[tid] = generate_color(tid)
            
            x, y, z = zip(*traj)
            self.lines[tid].set_data_3d(x, y, z)
            self.lines[tid].set_color(self.color_cache[tid])
            
            self.ax_3d.text(x[-1], y[-1], z[-1], str(tid), 
                          color=self.color_cache[tid], fontsize=8)

def main():
    processor = VideoProcessor()
    visualizer = Visualizer()
    processor.run(visualizer)

if __name__ == "__main__":
    main()