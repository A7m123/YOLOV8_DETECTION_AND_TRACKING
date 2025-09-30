import os
import logging

# Delay YOLO import to avoid threading issues
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        logging.error(f"Failed to import YOLO: {e}")
        raise e

# Flask integration variables
DEFAULT_MODEL_PATH = "yolov8l.pt"
DEFAULT_CONFIDENCE = 0.2
SUPPORTED_CLASSES = ["person", "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "keyboard", "cell phone"]  # Extended class list


class YoloDetector:
    def __init__(self, model_path, confidence):
        try:
            # Load YOLO dynamically
            YOLO = load_yolo()
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_path} not found, downloading...")
            
            # Initialize with GPU optimization
            self.model = YOLO(model_path)
            
            # Force GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    logging.info(f"YOLO model loaded on GPU: {torch.cuda.get_device_name()}")
                else:
                    logging.warning("CUDA not available, using CPU")
            except ImportError:
                logging.warning("PyTorch not available for GPU detection")
            
            self.classList = ["person", "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "keyboard", "cell phone"]
            self.confidence = confidence
            
            # Performance optimization settings
            self.model.overrides['verbose'] = False
            self.model.overrides['half'] = True  # Use FP16 for faster inference
            
            logging.info(f"YOLO model loaded successfully: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, image):
        try:
            # Add validation
            if image is None or image.size == 0:
                return []
            
            # Optimized prediction with GPU settings
            results = self.model.predict(
                image, 
                conf=self.confidence,
                verbose=False,
                half=True,  # Use FP16
                device='cuda' if self._is_cuda_available() else 'cpu',
                imgsz=640,  # Optimal input size
                max_det=100,  # Limit detections for speed
                agnostic_nms=True  # Faster NMS
            )
            
            if not results:
                return []
                
            result = results[0]
            detections = self.make_detections(result)
            return detections
            
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return []
    
    def _is_cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect_and_annotate(self, image):
        """Returns both detections and annotated image"""
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        detections = self.make_detections(result)
        annotated_image = result.plot()  # This creates the annotated image
        return detections, annotated_image

    def make_detections(self, result):
        try:
            boxes = result.boxes
            detections = []
            
            if boxes is None:
                return detections
                
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    class_number = int(box.cls[0])

                    if result.names[class_number] not in self.classList:
                        continue
                    conf = float(box.conf[0])
                    if conf < self.confidence:
                        continue
                    detections.append((([x1, y1, w, h]), class_number, conf))
                except Exception as e:
                    logging.error(f"Error processing detection box: {e}")
                    continue
                    
            return detections
        except Exception as e:
            logging.error(f"Error in make_detections: {e}")
            return []