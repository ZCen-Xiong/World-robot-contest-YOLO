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

class VoiceControlledObjectDetector:
    def __init__(self, lang='zh', model_path='yolov8n.pt'):
        self.recognizer = VoiceRecognizer(lang=lang)
        # self.detector = ObjectDetector(model_path=model_path)
        self.keywords = ['识字','十字','石子','一字','剪刀','捡到','六角']
        self.target_class = None
        self.class_map = {
            '识字': 'phillips',
            '十字': 'phillips',
            '石子': 'phillips',
            '一字': 'slotted',
            '剪刀': 'scissors',
            '捡到': 'scissors',
            '六角': 'hex',
            # 添加更多类别映射
        }

    def run_once(self):
        # 录音并识别命令
        audio_data = self.recognizer.record_audio()
        command = self.recognizer.recognize(audio_data)
        print("Recognized command:", command)
        recognized_object = self.extract_keyword(command)
        # target_name_cn = command.strip().lower()
        if recognized_object in self.class_map:
            target_name = self.class_map[recognized_object]
            print(f"Target class set to: {target_name}")
        else:
            print("Target class not recognized.")
            return
        return target_name
    
    def extract_keyword(self, command):
        for keyword in self.keywords:
            if keyword in command:
                return keyword
        return None


if __name__ == "__main__":
    detector = VoiceControlledObjectDetector(lang='zh')
    target = detector.run_once()
    # while 1:
    #     print(target)

    # rospy.init_node('voice_recognition_publisher', anonymous=True)
    # pub = rospy.Publisher('voice_commands', String, queue_size=10)
    # while not rospy.is_shutdown():
    #     rospy.loginfo(f"Publishing command: {command}")
    #     pub.publish(target)
    #     rate.sleep()
    
