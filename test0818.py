import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('space.pt')  # 使用yolov8n.pt模型，你可以根据需要替换为其他模型

# 设置RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始数据流
pipeline.start(config)

# 目标物体名称（你想检测的物体）
target_label = 'Lshape'  # 替换为你需要检测的物体名称

try:
    detected = False

    while True:
        # 获取图像帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 阶段一：物体检测
        results = model(color_image)
        for result in results:
            for box in result.boxes:
                label = box.label
                if label == target_label:
                    detected = True
                    print("目标物体已检测到，发送flag信号")
                    break
            if detected:
                break

        if detected:
            while True:
                # 再次获取图像帧
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # 阶段二：持续获取目标物体的坐标与深度信息
                results = model(color_image)
                for result in results:
                    for box in result.boxes:
                        label = box.label
                        if label == target_label:
                            x_center = int((box.xyxy[0] + box.xyxy[2]) / 2)
                            y_center = int((box.xyxy[1] + box.xyxy[3]) / 2)
                            depth_value = depth_image[y_center, x_center]

                            # 发送目标物体的坐标和深度值
                            print(f"物体坐标: ({x_center}, {y_center}), 深度值: {depth_value}")
                            break
                # 延迟几毫秒，避免占用过多资源
                cv2.waitKey(10)

except Exception as e:
    print(e)

finally:
    # 停止数据流
    pipeline.stop()
