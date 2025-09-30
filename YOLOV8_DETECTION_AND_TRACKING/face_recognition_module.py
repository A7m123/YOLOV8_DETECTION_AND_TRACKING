import cv2
import numpy as np
import os
import pickle
import logging
from datetime import datetime

# Try to import face_recognition, with fallback
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logging.info("face_recognition library loaded successfully")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition library not available. Face recognition features will be disabled.")

class FaceRecognitionManager:
    def __init__(self, faces_dir="saved_faces"):
        self.faces_dir = faces_dir
        self.known_faces = []
        self.known_names = []
        self.face_encodings_file = os.path.join(faces_dir, "face_encodings.pkl")
        self.enabled = FACE_RECOGNITION_AVAILABLE
        
        if not self.enabled:
            logging.warning("Face recognition manager initialized but disabled due to missing dependencies")
            return
        
        # Create faces directory if it doesn't exist
        os.makedirs(faces_dir, exist_ok=True)
        
        # Load existing face encodings
        self.load_face_encodings()
        
        logging.info(f"Face recognition initialized. Known faces: {len(self.known_faces)}")
    
    def detect_faces(self, image):
        """Detect faces in image and return face locations and encodings"""
        if not self.enabled:
            return [], []
            
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            return face_locations, face_encodings
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            return [], []
    
    def save_face(self, image, name):
        """Save a face with given name"""
        if not self.enabled:
            return False, "Face recognition not available - missing dependencies"
            
        try:
            face_locations, face_encodings = self.detect_faces(image)
            
            if not face_encodings:
                return False, "No face detected in the image"
            
            if len(face_encodings) > 1:
                return False, "Multiple faces detected. Please ensure only one face is visible"
            
            # Save face encoding
            face_encoding = face_encodings[0]
            face_location = face_locations[0]
            
            # Extract face image for storage
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            
            # Save face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"{name}_{timestamp}.jpg"
            face_path = os.path.join(self.faces_dir, face_filename)
            cv2.imwrite(face_path, face_image)
            
            # Add to known faces
            self.known_faces.append(face_encoding)
            self.known_names.append(name)
            
            # Save encodings to file
            self.save_face_encodings()
            
            logging.info(f"Face saved: {name}")
            return True, f"Face saved successfully as {name}"
            
        except Exception as e:
            logging.error(f"Error saving face: {e}")
            return False, f"Error saving face: {str(e)}"
    
    def recognize_faces(self, image):
        """Recognize faces in image and return results"""
        if not self.enabled:
            return []
            
        try:
            if not self.known_faces:
                return []
            
            face_locations, face_encodings = self.detect_faces(image)
            
            recognition_results = []
            
            for face_location, face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                name = "Unknown"
                confidence = 0.0
                
                if matches and len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]  # Convert distance to confidence
                
                recognition_results.append({
                    'location': face_location,
                    'name': name,
                    'confidence': confidence
                })
            
            return recognition_results
            
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return []
    
    def save_face_encodings(self):
        """Save face encodings to file"""
        if not self.enabled:
            return
            
        try:
            data = {
                'encodings': self.known_faces,
                'names': self.known_names
            }
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.error(f"Error saving face encodings: {e}")
    
    def load_face_encodings(self):
        """Load face encodings from file"""
        if not self.enabled:
            return
            
        try:
            if os.path.exists(self.face_encodings_file):
                with open(self.face_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                    logging.info(f"Loaded {len(self.known_faces)} saved faces")
        except Exception as e:
            logging.error(f"Error loading face encodings: {e}")
            self.known_faces = []
            self.known_names = []
    
    def delete_face(self, name):
        """Delete a saved face"""
        if not self.enabled:
            return False, "Face recognition not available"
            
        try:
            indices_to_remove = [i for i, n in enumerate(self.known_names) if n == name]
            
            if not indices_to_remove:
                return False, f"Face '{name}' not found"
            
            # Remove from lists (in reverse order to maintain indices)
            for i in reversed(indices_to_remove):
                self.known_faces.pop(i)
                self.known_names.pop(i)
            
            # Save updated encodings
            self.save_face_encodings()
            
            # Remove image files
            for filename in os.listdir(self.faces_dir):
                if filename.startswith(f"{name}_") and filename.endswith('.jpg'):
                    try:
                        os.remove(os.path.join(self.faces_dir, filename))
                    except Exception:
                        pass
            
            logging.info(f"Face deleted: {name}")
            return True, f"Face '{name}' deleted successfully"
            
        except Exception as e:
            logging.error(f"Error deleting face: {e}")
            return False, f"Error deleting face: {str(e)}"
    
    def get_saved_faces(self):
        """Get list of saved face names"""
        if not self.enabled:
            return []
        return list(set(self.known_names))  # Remove duplicates
