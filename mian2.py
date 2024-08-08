import pyrealsense2 as rs
import numpy as np
import pyaudio
import wave
from paddlespeech.cli.asr.infer import ASRExecutor
from ultralytics import YOLO
import cv2

class VoiceRecognizer:
    def __init__(self, lang='zh'):
        self.asr = ASRExecutor()
        self.lang = lang

    def record_audio(self, record_seconds=5, sample_rate=16000, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=1024)
        frames = []

        print("Recording...")
        for _ in range(0, int(sample_rate / 1024 * record_seconds)):
            data = stream.read(1024)
            frames.append(data)
        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = b''.join(frames)
        return audio_data

    def recognize(self, audio_data, sample_rate=16000):
        temp_wav_file = "temp.wav"
        with wave.open(temp_wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        
        try:
            result = self.asr(temp_wav_file)
            return result
        except Exception as e:
            print(f"Error during ASR inference: {e}")
            return ""

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # 获取类别名称映射

    def detect_objects(self, frame):
        results = self.model(frame)
        return results

    def get_coordinates(self, results, target_class):
        coordinates = []
        try:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 盒子的坐标
            class_ids = results[0].boxes.cls.cpu().numpy()  # 类别ID

            for box, class_id in zip(boxes, class_ids):
                if int(class_id) == target_class:
                    x1, y1, x2, y2 = box
                    coordinates.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception as e:
            print(f"Error: Failed to parse YOLO model results: {e}")
        return coordinates

    def get_object_names(self, results):
        try:
            class_ids = results[0].boxes.cls.cpu().numpy()  # 类别ID
            object_names = [self.class_names[int(class_id)] for class_id in class_ids]
            return object_names
        except Exception as e:
            print(f"Error: Failed to get object names: {e}")
            return []

    def draw_boxes(self, frame, coordinates):
        for (x1, y1, x2, y2) in coordinates:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

class VoiceControlledObjectDetector:
    def __init__(self, lang='zh', model_path='yolov8n.pt'):
        self.recognizer = VoiceRecognizer(lang=lang)
        self.detector = ObjectDetector(model_path=model_path)
        self.target_class = None
        self.class_map = {
            '鼠标': 'mouse',
            '键盘': 'keyboard'
            # 添加更多类别映射
        }

    def run_once(self):
        # 录音并识别命令
        # audio_data = self.recognizer.record_audio()
        # command = self.recognizer.recognize(audio_data)
        # print("Recognized command:", command)

        # target_name_cn = command.strip().lower()
        # if target_name_cn in self.class_map:
        #     target_name = self.class_map[target_name_cn]
        #     print(f"Target class set to: {target_name}")
        # else:
        #     print("Target class not recognized.")
        #     return
        target_name = 'mouse'

        # 配置 RealSense 管道
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 开始流
        pipeline.start(config)

        try:
            # 获取一帧深度和颜色图像
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Failed to capture frames.")
                return

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 目标检测
            results = self.detector.detect_objects(color_image)
            object_names = self.detector.get_object_names(results)
            print("Detected objects:", object_names)

            if target_name in object_names:
                target_class_id = [k for k, v in self.detector.class_names.items() if v == target_name][0]
                coordinates = self.detector.get_coordinates(results, target_class=target_class_id)

                for (x1, y1, x2, y2) in coordinates:
                    # 获取中心点
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    distance = depth_frame.get_distance(x_center, y_center)
                    print(f"Detected {target_name} at coordinates: ({x_center}, {y_center}, {distance}m)")

                    # 在图像上绘制框
                    color_image = self.detector.draw_boxes(color_image, [(x1, y1, x2, y2)])

            # 显示检测结果
            cv2.imshow('YOLOv8 Detection', color_image)
            cv2.waitKey(0)  # 等待按键按下以关闭窗口

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = VoiceControlledObjectDetector(lang='zh')
    detector.run_once()

    # detector = ObjectDetector

