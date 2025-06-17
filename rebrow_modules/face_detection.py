import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Any

# Import MediaPipe solutions explicitly
mp_face_mesh = mp.solutions.face_mesh # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore

class FaceDetector:
    def __init__(self, use_gpu=True):
        # Create FaceMesh instance with GPU support if available
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe indices for eyebrows - using standard indices that work reliably
        # Left eyebrow indices
        self.left_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        # Right eyebrow indices
        self.right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        
    def detect_face(self, image: np.ndarray) -> Any:
        """Detect face in the image and return landmarks"""
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get face landmarks
        # Set image to writeable=False to improve performance
        rgb_image.flags.writeable = False
        results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True
        
        return results
    
    def crop_face(self, image: np.ndarray, results: Any, padding: float = 0.2) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Crop the image to the face area with padding"""
        if not results.multi_face_landmarks:
            return image, None
        
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Find bounding box of face
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0
        
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add padding
        padding_x = int((x_max - x_min) * padding)
        padding_y = int((y_max - y_min) * padding)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        # Crop the image
        face_crop = image[y_min:y_max, x_min:x_max]
        
        # Return cropped image and crop coordinates
        return face_crop, (x_min, y_min, x_max, y_max)
    
    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
        """Draw all facial landmarks on the image"""
        if not results.multi_face_landmarks:
            return image
        
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw the face contours
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw the eyebrows specifically
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        return annotated_image
    
    def get_eyebrow_landmarks(self, image: np.ndarray, results: Any, crop_coords: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
        """Extract eyebrow landmarks from face landmarks"""
        if not results.multi_face_landmarks:
            return None, None
        
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        left_eyebrow = []
        right_eyebrow = []
        
        # Extract left eyebrow landmarks
        for idx in self.left_eyebrow_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            
            # Adjust coordinates if we're working with a cropped image
            if crop_coords is not None:
                x_min, y_min, _, _ = crop_coords
                x = x - x_min
                y = y - y_min
                
            left_eyebrow.append((x, y))
        
        # Extract right eyebrow landmarks
        for idx in self.right_eyebrow_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            
            # Adjust coordinates if we're working with a cropped image
            if crop_coords is not None:
                x_min, y_min, _, _ = crop_coords
                x = x - x_min
                y = y - y_min
                
            right_eyebrow.append((x, y))
        
        # Ensure we have enough landmarks for proper detection
        if len(left_eyebrow) < 5 or len(right_eyebrow) < 5:
            return None, None
            
        return left_eyebrow, right_eyebrow
    
    def draw_eyebrow_landmarks(self, image: np.ndarray, left_eyebrow: Optional[List[Tuple[int, int]]], right_eyebrow: Optional[List[Tuple[int, int]]]) -> np.ndarray:
        """Draw eyebrow landmarks on the image"""
        if left_eyebrow is None or right_eyebrow is None:
            return image
        
        annotated_image = image.copy()
        
        # Draw left eyebrow landmarks
        for point in left_eyebrow:
            cv2.circle(annotated_image, point, 2, (0, 0, 255), -1)
        
        # Draw right eyebrow landmarks
        for point in right_eyebrow:
            cv2.circle(annotated_image, point, 2, (255, 0, 0), -1)
        
        # Connect points to show eyebrow shape
        if len(left_eyebrow) > 1:
            cv2.polylines(annotated_image, [np.array(left_eyebrow)], False, (0, 0, 255), 1)
        
        if len(right_eyebrow) > 1:
            cv2.polylines(annotated_image, [np.array(right_eyebrow)], False, (255, 0, 0), 1)
        
        return annotated_image
