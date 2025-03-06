import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
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

# 初始化MediaPipe绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def generate_color(track_id):
    """生成基于track_id的固定颜色"""
    golden_ratio = 0.618033988749895
    hue = (hash(track_id) * golden_ratio) % 1.0
    return plt.cm.hsv([hue])[0][:3]

class VideoProcessor:
    def __init__(self):
        self.detector = None
        self.pose_estimator = None
        self.cap = None
        self.trajectories = defaultdict(list)
        self.kalman_filters = dict()
        self.frame_size = None
        self.track_lifespan = defaultdict(int)
        self.out_video = None
        self.last_processed_frame = None  # 保存最后一帧用于填充

    def create_kf(self):
        """初始化卡尔曼滤波器"""
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
        """视频处理核心逻辑"""
        start_time = time.time()
        results = self.detector.track(frame, persist=True, iou=0.7, verbose=False)
        print(f"Detection time: {time.time() - start_time:.2f} seconds")
        
        output_data = {"frame": None, "trajectories": {}}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                margin = 0
                h, w = frame.shape[:2]
                roi = frame[max(y1+margin, 0):min(y2-margin, h), max(x1+margin, 0):min(x2-margin, w)]

                cv2.imwrite(f"roi_{tid}.jpg", roi)

                if roi.size == 0: continue

                pose_results = self.pose_estimator.process(
                    cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                
                if pose_results.pose_landmarks:
                    self.update_trajectory(tid, pose_results.pose_landmarks, frame.shape)
                    mp_drawing.draw_landmarks(
                        roi,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

                    left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    hip_center_x = int((left_hip.x + right_hip.x) / 2 * (x2 - x1 - 2 * margin) + x1 + margin)
                    hip_center_y = int((left_hip.y + right_hip.y) / 2 * (y2 - y1 - 2 * margin) + y1 + margin)

                    cv2.circle(frame, (hip_center_x, hip_center_y), 5, (0, 0, 255), -1)
                    text = "center of hip"
                    text_position = (hip_center_x + 10, hip_center_y + 10)
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

                else:
                    if tid in self.kalman_filters:
                        kf = self.kalman_filters[tid]
                        kf.predict()
                        smoothed_pos = kf.x[:3].tolist()
                        self.trajectories[tid].append(smoothed_pos)
                    print(f"Track ID {tid}: No pose landmarks detected.")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        self.last_processed_frame = frame  # 保存最后一帧
        return output_data

    def update_trajectory(self, tid, landmarks, frame_shape):
        """更新3D轨迹"""
        h, w = frame_shape[:2]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        x = (left_hip.x + right_hip.x) / 2 * w
        y = (left_hip.y + right_hip.y) / 2 * h
        z = (left_hip.z + right_hip.z) / 2 * w * 2.0

        if tid not in self.trajectories:
            self.trajectories[tid] = []
        self.trajectories[tid].append([x, y, z])
        if len(self.trajectories[tid]) > 10:
            smoothed_pos = np.mean(self.trajectories[tid][-10:], axis=0)
        else:
            smoothed_pos = [x, y, z]

        if tid not in self.kalman_filters:
            self.kalman_filters[tid] = self.create_kf()
            self.kalman_filters[tid].x = np.array([smoothed_pos[0], smoothed_pos[1], smoothed_pos[2], 0, 0, 0])
        
        kf = self.kalman_filters[tid]
        kf.predict()
        kf.update(smoothed_pos)
        
        smoothed_pos = kf.x[:3].tolist()
        self.trajectories[tid][-1] = smoothed_pos
        if len(self.trajectories[tid]) > 50:
            self.trajectories[tid].pop(0)

    def initialize(self):
        """初始化资源"""
        model_path = 'yolov8m.pt'
        if os.path.exists(model_path):
            print(f"Loading local model from {model_path}")
            self.detector = YOLO(model_path)
        else:
            print("Local model not found, downloading yolov8m.pt")
            self.detector = YOLO('yolov8m.pt')

        self.pose_estimator = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        
        video_path = "7.mp4"
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video loaded: {self.frame_size}, {self.total_frames} frames, {self.fps:.2f} FPS")

        # 初始化视频输出
        output_path = "output.mp4"
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
            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    print("Error: Failed to pre-read video frames")
                    stop_event.set()
                    return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_count = 0
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print(f"End of video reached after {frame_count} frames")
                    stop_event.set()
                    break
                frame_count += 1
                print(f"Read frame {frame_count}")
                try:
                    frame_queue.put(frame, timeout=1.0)
                except queue.Full:
                    print(f"Frame {frame_count} dropped: frame_queue full")
                time.sleep(1.0 / self.fps)  # 匹配输入视频帧率

        def process_frames():
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
        saved_frame_count = 0
        last_time = time.time()
        while saved_frame_count < self.total_frames:  # 确保保存所有帧
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
                saved_frame_count += 1

                current_time = time.time()
                elapsed = current_time - last_time
                sleep_time = max(0.033 - elapsed, 0)
                time.sleep(sleep_time)
                last_time = current_time
                processed_queue.task_done()
            except queue.Empty:
                # 如果队列为空但未保存足够帧数，用最后一帧填充
                if self.last_processed_frame is not None and saved_frame_count < self.total_frames:
                    self.out_video.write(self.last_processed_frame)
                    saved_frame_count += 1
                    print(f"Filling frame {saved_frame_count} with last processed frame")
                time.sleep(0.033)

        t1.join()
        t2.join()

        print("Video processing complete. Window will close in 5 seconds.")
        time.sleep(5)
        self.cap.release()
        self.out_video.release()
        self.pose_estimator.close()
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

