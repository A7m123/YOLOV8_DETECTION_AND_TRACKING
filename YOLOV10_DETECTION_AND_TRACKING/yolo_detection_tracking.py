import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "yolov8l.pt"  # Path to the YOLOv8 model
VIDEO_PATH = "assets/football.mp4"  # Path to the input video file

# Flask integration variables
USE_FLASK = False  # Set to True to disable standalone mode
FLASK_HOST = "0.0.0.0"  # Flask server host
FLASK_PORT = 5000  # Flask server port

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(
                bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{str(tracking_id)}", (int(bounding_box[0]), int(
                bounding_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        cv2.imshow("YOLOv8L Detection and Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
