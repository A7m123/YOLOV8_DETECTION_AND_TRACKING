import eventlet
eventlet.monkey_patch()

# System imports only (no third-party threading libraries yet)
import base64
import numpy as np
from datetime import datetime
import logging
import queue
import threading
import time
from collections import defaultdict
import uuid
import sys
import os

# Flask imports (after monkey patch)
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# Configure logging early
logging.basicConfig(level=logging.INFO)

# Reduce logging
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('engineio.server').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables for queue management - OPTIMIZED
frame_queues = defaultdict(queue.Queue)
processing_threads = {}
client_status = {}
MAX_QUEUE_SIZE = 20  # Increased for better throughput
PROCESSING_WORKERS = 6  # Increased workers for better parallelization
BATCH_SIZE = 2  # Process frames in batches when possible

# Global variables for YOLO modules
YOLO_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False
YoloDetector = None
Tracker = None
FaceRecognitionManager = None
cv2 = None

def load_cv2():
    """Load OpenCV safely after monkey patching"""
    global cv2
    try:
        import cv2 as cv2_module
        cv2 = cv2_module
        return True
    except ImportError as e:
        logging.error(f"OpenCV not available: {e}")
        return False

def load_yolo_modules():
    """Load YOLO modules safely after monkey patching"""
    global YOLO_AVAILABLE, YoloDetector, Tracker, FaceRecognitionManager, FACE_RECOGNITION_AVAILABLE
    try:
        # Import YOLO modules after monkey patch and cv2
        from yolo_detector import YoloDetector
        from tracker import Tracker
        from face_recognition_module import FaceRecognitionManager
        YOLO_AVAILABLE = True
        FACE_RECOGNITION_AVAILABLE = True
    except ImportError as e:
        logging.error(f"YOLO modules not available: {e}")
        YOLO_AVAILABLE = False
        FACE_RECOGNITION_AVAILABLE = False
    except Exception as e:
        logging.error(f"Error loading YOLO modules: {e}")
        YOLO_AVAILABLE = False
        FACE_RECOGNITION_AVAILABLE = False

class FrameProcessor:
    def __init__(self):
        self.active = True
        self.lock = threading.Lock()
        self.yolo_enabled = False
        self.detector = None
        self.tracker = None
        self.face_recognition = None
        
        # Load OpenCV first
        if not load_cv2():
            logging.error("Cannot proceed without OpenCV")
            return
        
        # Try to initialize YOLO detector and tracker with GPU optimization
        if load_yolo_modules() and YOLO_AVAILABLE:
            try:
                # Enhanced YOLO initialization with GPU optimization
                self.detector = YoloDetector("yolov8l.pt", confidence=0.25)
                self.tracker = Tracker()
                self.yolo_enabled = True
                logging.info("YOLO detection and tracking initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize YOLO: {e}")
                self.yolo_enabled = False
                self.detector = None
                self.tracker = None
        else:
            logging.warning("YOLO not available, running without detection")
        
        # Initialize face recognition
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognition = FaceRecognitionManager()
                logging.info("Face recognition initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize face recognition: {e}")
                self.face_recognition = None
    
    def process_frame(self, frame, client_id):
        """
        Process a single frame with optimized YOLO detection, tracking, and face recognition
        """
        try:
            # Start timing for performance monitoring
            start_time = time.time()
            
            # Validate frame
            if frame is None or frame.size == 0:
                logging.error(f"Invalid frame received from {client_id}")
                return None
            
            # Optimized resize - use different sizes based on load
            current_queue_size = frame_queues[client_id].qsize()
            
            # Dynamic resolution based on queue size for better performance
            if current_queue_size > 15:
                # High load - use smaller resolution
                new_width = 480
            elif current_queue_size > 10:
                # Medium load - use medium resolution
                new_width = 560
            else:
                # Low load - use higher resolution
                new_width = 640
                
            height, width = frame.shape[:2]
            new_height = int((new_width / width) * height)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # YOLO Detection and Tracking with optimizations
            detection_info = ""
            object_count = 0
            
            if self.yolo_enabled and self.detector and self.tracker:
                try:
                    # Detect objects with optimized settings
                    detections = self.detector.detect(frame)
                    
                    # Track objects
                    tracking_ids, boxes = self.tracker.track(detections, frame)
                    object_count = len(tracking_ids)
                    
                    # Draw bounding boxes and tracking IDs
                    for tracking_id, bounding_box in zip(tracking_ids, boxes):
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, int(bounding_box[0]))
                        y1 = max(0, int(bounding_box[1]))
                        x2 = min(frame.shape[1], int(bounding_box[2]))
                        y2 = min(frame.shape[0], int(bounding_box[3]))
                        
                        # Draw bounding box with thicker line for visibility
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw tracking ID with background
                        label = f"ID:{str(tracking_id)}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    detection_info = f"Objects: {object_count}"
                    
                except Exception as e:
                    logging.error(f"YOLO processing error: {e}")
                    detection_info = "YOLO Error"
            else:
                detection_info = "YOLO Disabled"
            
            # Face Recognition
            face_info = ""
            recognized_faces = []
            if self.face_recognition:
                try:
                    recognized_faces = self.face_recognition.recognize_faces(frame)
                    
                    # Draw face recognition results
                    for face_result in recognized_faces:
                        top, right, bottom, left = face_result['location']
                        name = face_result['name']
                        confidence = face_result['confidence']
                        
                        # Draw face rectangle
                        color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)  # Yellow for known, red for unknown
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw name and confidence
                        if name != "Unknown":
                            label = f"{name} ({confidence:.2f})"
                        else:
                            label = "Unknown"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (left, bottom), (left + label_size[0], bottom + label_size[1] + 10), color, -1)
                        cv2.putText(frame, label, (left, bottom + label_size[1] + 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    if recognized_faces:
                        known_faces = [f['name'] for f in recognized_faces if f['name'] != "Unknown"]
                        unknown_count = len([f for f in recognized_faces if f['name'] == "Unknown"])
                        
                        face_info = f"Faces: {len(known_faces)} known"
                        if unknown_count > 0:
                            face_info += f", {unknown_count} unknown"
                
                except Exception as e:
                    logging.error(f"Face recognition error: {e}")
                    face_info = "Face recognition error"
            
            # Add performance info with better positioning
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            processing_time = (time.time() - start_time) * 1000
            queue_size = current_queue_size
            
            # Create info overlay with semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text information
            info_lines = [
                f'Time: {timestamp}',
                f'Client: {client_id[:8]}',
                f'Processing: {processing_time:.1f}ms',
                f'Queue: {queue_size}/{MAX_QUEUE_SIZE}',
                f'{detection_info}',
                f'{face_info}',
                f'Resolution: {new_width}x{new_height}'
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 25 + (i * 20)
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 255), 1)
            
            return frame, object_count, recognized_faces
            
        except Exception as e:
            logging.error(f"Error processing frame for {client_id}: {e}")
            try:
                cv2.putText(frame, "Processing Error", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                return frame, 0, []
            except:
                return frame, 0, []

class OptimizedProcessingWorker(threading.Thread):
    def __init__(self, worker_id, processor):
        super().__init__()
        self.worker_id = worker_id
        self.processor = processor
        self.daemon = True
        self.performance_stats = {
            'frames_processed': 0,
            'avg_processing_time': 0,
            'last_reset': time.time()
        }
        
    def run(self):
        logging.info(f"Starting optimized processing worker {self.worker_id}")
        while self.processor.active:
            try:
                frames_processed_this_cycle = 0
                
                # Check all client queues with priority handling
                client_list = list(frame_queues.keys())
                
                # Sort clients by queue size (prioritize fuller queues)
                client_list.sort(key=lambda cid: frame_queues[cid].qsize(), reverse=True)
                
                for client_id in client_list:
                    try:
                        # Process multiple frames per client if queue is full
                        frames_to_process = min(2, frame_queues[client_id].qsize())
                        
                        for _ in range(frames_to_process):
                            if frame_queues[client_id].empty():
                                break
                                
                            frame_data = frame_queues[client_id].get_nowait()
                            
                            # Process the frame
                            start_process_time = time.time()
                            result = self.processor.process_frame(
                                frame_data['frame'], 
                                client_id
                            )
                            
                            if result is None:
                                frame_queues[client_id].task_done()
                                continue
                                
                            processed_frame, object_count, recognized_faces = result
                            
                            # Convert to base64 with optimized quality
                            quality = 90 if frame_queues[client_id].qsize() < 5 else 80
                            success, buffer = cv2.imencode('.jpg', processed_frame, [
                                cv2.IMWRITE_JPEG_QUALITY, quality
                            ])
                            
                            if not success:
                                frame_queues[client_id].task_done()
                                continue
                                
                            processed_image_data = base64.b64encode(buffer).decode('utf-8')
                            
                            # Update performance stats
                            process_time = (time.time() - start_process_time) * 1000
                            self.performance_stats['frames_processed'] += 1
                            self.performance_stats['avg_processing_time'] = (
                                (self.performance_stats['avg_processing_time'] * 
                                 (self.performance_stats['frames_processed'] - 1) + process_time) /
                                self.performance_stats['frames_processed']
                            )
                            
                            # Emit back to client with face recognition data
                            socketio.emit('processed_frame', {
                                'image': f'data:image/jpeg;base64,{processed_image_data}',
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'worker_id': self.worker_id,
                                'queue_size': frame_queues[client_id].qsize(),
                                'detected_objects': object_count,
                                'recognized_faces': recognized_faces,
                                'processing_time': round(process_time, 1),
                                'worker_performance': round(self.performance_stats['avg_processing_time'], 1)
                            }, room=client_id)
                            
                            frame_queues[client_id].task_done()
                            frames_processed_this_cycle += 1
                            
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logging.error(f"Worker {self.worker_id} error for {client_id}: {e}")
                
                # Dynamic sleep based on workload
                if frames_processed_this_cycle == 0:
                    time.sleep(0.005)  # Longer sleep when idle
                elif frames_processed_this_cycle < 3:
                    time.sleep(0.001)  # Short sleep when moderate load
                # No sleep when high load
                
                # Reset performance stats every minute
                if time.time() - self.performance_stats['last_reset'] > 60:
                    self.performance_stats = {
                        'frames_processed': 0,
                        'avg_processing_time': 0,
                        'last_reset': time.time()
                    }
                
            except Exception as e:
                logging.error(f"Worker {self.worker_id} general error: {e}")
                time.sleep(0.1)

# Initialize processor and workers with optimizations
frame_processor = FrameProcessor()
worker_threads = []

def start_workers():
    """Start the optimized worker threads"""
    for i in range(PROCESSING_WORKERS):
        worker = OptimizedProcessingWorker(i, frame_processor)
        worker.start()
        worker_threads.append(worker)
    logging.info(f"Started {PROCESSING_WORKERS} optimized processing workers")

def cleanup_client(client_id):
    """Clean up resources for a disconnected client"""
    if client_id in frame_queues:
        # Clear the queue
        while not frame_queues[client_id].empty():
            try:
                frame_queues[client_id].get_nowait()
                frame_queues[client_id].task_done()
            except queue.Empty:
                break
        del frame_queues[client_id]
    
    if client_id in client_status:
        del client_status[client_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    """Endpoint to check server status"""
    status_info = {
        'active_clients': len(frame_queues),
        'total_workers': len(worker_threads),
        'queue_sizes': {cid: q.qsize() for cid, q in frame_queues.items()},
        'client_status': client_status
    }
    return status_info

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    client_status[client_id] = {
        'connected_at': datetime.now().isoformat(),
        'frames_processed': 0,
        'last_activity': datetime.now().isoformat()
    }
    logging.info(f'Client connected: {client_id}')
    emit('connection_response', {
        'data': 'Connected to server',
        'client_id': client_id,
        'workers': PROCESSING_WORKERS
    })

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    logging.info(f'Client disconnected: {client_id}')
    cleanup_client(client_id)

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Receive frame from client and add to processing queue with optimizations
    """
    client_id = request.sid
    
    try:
        # Check if cv2 is available
        if cv2 is None:
            emit('error', {'message': 'OpenCV not available on server'}, room=client_id)
            return
            
        # Update client activity
        client_status[client_id]['last_activity'] = datetime.now().isoformat()
        
        # Advanced queue management with dynamic thresholds
        current_queue_size = frame_queues[client_id].qsize()
        
        # Dynamic queue size management
        if current_queue_size >= MAX_QUEUE_SIZE:
            # Clear some old frames to make room
            frames_to_remove = min(5, current_queue_size - MAX_QUEUE_SIZE + 2)
            for _ in range(frames_to_remove):
                try:
                    frame_queues[client_id].get_nowait()
                    frame_queues[client_id].task_done()
                except queue.Empty:
                    break
            
            logging.warning(f"Queue management for {client_id}: removed {frames_to_remove} frames")
            
        elif current_queue_size >= MAX_QUEUE_SIZE * 0.8:  # 80% threshold
            # Skip this frame to prevent queue overflow
            emit('queue_warning', {
                'message': f'High queue load ({current_queue_size}/{MAX_QUEUE_SIZE}), optimizing...',
                'queue_size': current_queue_size,
                'action': 'frame_skipped'
            }, room=client_id)
            return
        
        # Validate input data
        if 'image' not in data or not data['image']:
            return
            
        # Extract and decode image
        try:
            image_parts = data['image'].split(',')
            if len(image_parts) != 2:
                return
                
            image_data = image_parts[1]
            if not image_data:
                return
            
            # Add padding if necessary
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
            
            # Decode base64
            try:
                img_bytes = base64.b64decode(image_data)
            except Exception:
                return
                
            if not img_bytes:
                return
                
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            if nparr.size == 0:
                return
                
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        except Exception as decode_error:
            logging.error(f"Image decode error for {client_id}: {decode_error}")
            return
        
        if frame is None or frame.size == 0:
            return
            
        # Add to processing queue with priority
        frame_data = {
            'frame': frame,
            'timestamp': datetime.now(),
            'client_id': client_id,
            'priority': 1 if current_queue_size < 5 else 0  # Higher priority for low-load clients
        }
        
        frame_queues[client_id].put(frame_data)
        client_status[client_id]['frames_processed'] += 1
        
    except Exception as e:
        logging.error(f"Error handling frame for {client_id}: {e}")

@socketio.on('client_stats')
def handle_client_stats():
    """Send statistics back to client"""
    client_id = request.sid
    if client_id in client_status:
        stats = client_status[client_id].copy()
        stats['queue_size'] = frame_queues[client_id].qsize() if client_id in frame_queues else 0
        emit('server_stats', stats)

def process_frame(frame, client_id):
    """Legacy function for backward compatibility"""
    return frame_processor.process_frame(frame, client_id)

# Add face recognition endpoints
@socketio.on('save_face')
def handle_save_face(data):
    """Save a face for recognition"""
    client_id = request.sid
    
    try:
        if not frame_processor.face_recognition:
            emit('face_save_response', {
                'success': False,
                'message': 'Face recognition not available'
            }, room=client_id)
            return
        
        # Get the current frame or use provided image
        if 'image' not in data or 'name' not in data:
            emit('face_save_response', {
                'success': False,
                'message': 'Image and name required'
            }, room=client_id)
            return
        
        name = data['name'].strip()
        if not name:
            emit('face_save_response', {
                'success': False,
                'message': 'Name cannot be empty'
            }, room=client_id)
            return
        
        # Decode image
        try:
            image_parts = data['image'].split(',')
            if len(image_parts) != 2:
                raise ValueError("Invalid image format")
            
            image_data = image_parts[1]
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode image")
                
        except Exception as e:
            emit('face_save_response', {
                'success': False,
                'message': f'Invalid image data: {str(e)}'
            }, room=client_id)
            return
        
        # Save face
        success, message = frame_processor.face_recognition.save_face(frame, name)
        
        emit('face_save_response', {
            'success': success,
            'message': message,
            'saved_faces': frame_processor.face_recognition.get_saved_faces()
        }, room=client_id)
        
    except Exception as e:
        logging.error(f"Error saving face for {client_id}: {e}")
        emit('face_save_response', {
            'success': False,
            'message': f'Error saving face: {str(e)}'
        }, room=client_id)

@socketio.on('get_saved_faces')
def handle_get_saved_faces():
    """Get list of saved faces"""
    client_id = request.sid
    
    try:
        if frame_processor.face_recognition:
            saved_faces = frame_processor.face_recognition.get_saved_faces()
        else:
            saved_faces = []
            
        emit('saved_faces_list', {
            'faces': saved_faces
        }, room=client_id)
        
    except Exception as e:
        logging.error(f"Error getting saved faces for {client_id}: {e}")
        emit('saved_faces_list', {'faces': []}, room=client_id)

@socketio.on('delete_face')
def handle_delete_face(data):
    """Delete a saved face"""
    client_id = request.sid
    
    try:
        if not frame_processor.face_recognition:
            emit('face_delete_response', {
                'success': False,
                'message': 'Face recognition not available'
            }, room=client_id)
            return
        
        if 'name' not in data:
            emit('face_delete_response', {
                'success': False,
                'message': 'Name required'
            }, room=client_id)
            return
        
        name = data['name']
        success, message = frame_processor.face_recognition.delete_face(name)
        
        emit('face_delete_response', {
            'success': success,
            'message': message,
            'saved_faces': frame_processor.face_recognition.get_saved_faces()
        }, room=client_id)
        
    except Exception as e:
        logging.error(f"Error deleting face for {client_id}: {e}")
        emit('face_delete_response', {
            'success': False,
            'message': f'Error deleting face: {str(e)}'
        }, room=client_id)

if __name__ == '__main__':
    # Start worker threads
    start_workers()
    
    print("=" * 60)
    print("OPTIMIZED Flask-SocketIO Camera Server with YOLO")
    print(f"Processing workers: {PROCESSING_WORKERS}")
    print(f"Max queue size: {MAX_QUEUE_SIZE}")
    print(f"Batch processing: {BATCH_SIZE}")
    print(f"OpenCV available: {cv2 is not None}")
    print(f"YOLO available: {YOLO_AVAILABLE}")
    print("Access the application at: http://localhost:5000")
    print("Server status at: http://localhost:5000/status")
    print("=" * 60)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, log_output=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        frame_processor.active = False
        for worker in worker_threads:
            worker.join(timeout=2)
        print("Server shutdown complete")