from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model(frame)
        return results

    def draw_boxes(self, frame, results, target_class):
        for result in results:
            for box in result.boxes:
                cls = box.cls
                conf = box.conf
                if cls == target_class:
                    x1, y1, x2, y2 = box.xyxy
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{target_class} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

# 示例用法
if __name__ == "__main__":
    detector = ObjectDetector(model_path='yolov8n.pt')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.detect_objects(frame)
        frame = detector.draw_boxes(frame, results, target_class=0)  # 假设目标类的ID为0

        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
