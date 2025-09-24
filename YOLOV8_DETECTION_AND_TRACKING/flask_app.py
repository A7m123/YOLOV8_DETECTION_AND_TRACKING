from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
from yolo_detector import YoloDetector
from tracker import Tracker
import threading
import queue

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = None
tracker = None
frame_queue = queue.Queue(maxsize=10)
processing_active = False

def initialize_models():
    global detector, tracker
    detector = YoloDetector(model_path="yolov8l.pt", confidence=0.2)
    tracker = Tracker()

def process_frame_worker():
    global processing_active
    while processing_active:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get(timeout=1)
                
                start_time = time.perf_counter()
                detections = detector.detect(frame)
                tracking_ids, boxes = tracker.update(detections, frame)
                
                # Draw annotations
                for tracking_id, bounding_box in zip(tracking_ids, boxes):
                    cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                                (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"{str(tracking_id)}", 
                              (int(bounding_box[0]), int(bounding_box[1] - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                end_time = time.perf_counter()
                fps = 1 / (end_time - start_time)
                
                # Add FPS to frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Emit processed frame
                socketio.emit('processed_frame', {'frame': frame_base64})
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error processing frame: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_processing')
def handle_start_processing():
    global processing_active
    if not processing_active:
        initialize_models()
        processing_active = True
        threading.Thread(target=process_frame_worker, daemon=True).start()
        emit('status', {'msg': 'Processing started'})

@socketio.on('stop_processing')
def handle_stop_processing():
    global processing_active
    processing_active = False
    emit('status', {'msg': 'Processing stopped'})

@socketio.on('frame_data')
def handle_frame(data):
    try:
        # Decode base64 frame
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Add frame to processing queue
        if not frame_queue.full():
            frame_queue.put(frame)
        
    except Exception as e:
        print(f"Error handling frame: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
