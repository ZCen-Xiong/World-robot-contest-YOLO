import pyaudio
import wave
from paddlespeech.cli.asr.infer import ASRExecutor

class VoiceRecognizer:
    def __init__(self, lang='en'):
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

# 示例用法
if __name__ == "__main__":
    recognizer = VoiceRecognizer(lang='en')  # 'en' for English, 'zh' for Chinese
    audio_data = recognizer.record_audio()
    text = recognizer.recognize(audio_data)
    print("Recognized text:", text)
