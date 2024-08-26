import pyaudio
import wave
from paddlespeech.cli.asr.infer import ASRExecutor
from ultralytics import YOLO
import cv2

class VoiceRecognizer:
    def __init__(self, lang='zh'):
        self.lang = lang
        if self.lang == 'en':
            self.asr = ASRExecutor(model='conformer_wenetspeech-en-16k')
        else:
            self.asr = ASRExecutor(model='conformer_wenetspeech-zh-16k')

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
        for result in results:
            for box in result.boxes:
                if box.cls == target_class:
                    x1, y1, x2, y2 = box.xyxy
                    coordinates.append((int(x1), int(y1), int(x2), int(y2)))
        return coordinates

    def get_object_names(self, results):
        class_ids = results[0].boxes.cls.cpu().numpy()  # 类别ID
        object_names = [self.class_names[int(class_id)] for class_id in class_ids]
        return object_names

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
        audio_data = self.recognizer.record_audio()
        command = self.recognizer.recognize(audio_data)
        print("Recognized command:", command)

        target_name_cn = command.strip().lower()
        if target_name_cn in self.class_map:
            target_name = self.class_map[target_name_cn]
            print(f"Target class set to: {target_name}")
        else:
            print("Target class not recognized.")
            return

        # 获取摄像头画面并检测目标物体
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            return

        results = self.detector.detect_objects(frame)
        object_names = self.detector.get_object_names(results)
        print("Detected objects:", object_names)

        if target_name in object_names:
            target_class_id = [k for k, v in self.detector.class_names.items() if v == target_name][0]
            coordinates = self.detector.get_coordinates(results, target_class=target_class_id)
            for coord in coordinates:
                print(f"Detected {target_name} at coordinates: {coord}")
            frame = self.detector.draw_boxes(frame, coordinates)

        # 显示检测结果
        # cv2.imshow('YOLOv8 Detection', frame)
        # cv2.waitKey(0)  # 等待按键按下以关闭窗口

        # cap.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = VoiceControlledObjectDetector(lang='zh')
    detector.run_once()
