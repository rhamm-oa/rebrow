import cv2
import numpy as np
import math

class ShapeAnalysis:
    def __init__(self):
        pass
    
    def analyze_eyebrow_shape(self, landmarks):
        """Analyze the shape of the eyebrow based on landmarks"""
        if not landmarks or len(landmarks) < 3:
            return None
        
        # Convert landmarks to numpy array
        points = np.array(landmarks, dtype=np.int32)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate area of the eyebrow
        area = cv2.contourArea(points)
        
        # Calculate perimeter of the eyebrow
        perimeter = cv2.arcLength(points, True)
        
        # Calculate compactness (circularity)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculate convex hull
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        
        # Calculate convexity (area / hull_area)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Calculate the arch of the eyebrow
        # Sort points by x-coordinate
        sorted_points = sorted(landmarks, key=lambda p: p[0])
        
        # Take points at 25%, 50%, and 75% positions
        p1 = sorted_points[len(sorted_points) // 4]
        p2 = sorted_points[len(sorted_points) // 2]
        p3 = sorted_points[3 * len(sorted_points) // 4]
        
        # Calculate the arch height (distance from middle point to line connecting first and last points)
        # Line equation: ax + by + c = 0
        a = p3[1] - p1[1]
        b = p1[0] - p3[0]
        c = p3[0] * p1[1] - p1[0] * p3[1]
        
        # Distance from point to line
        arch_height = abs(a * p2[0] + b * p2[1] + c) / math.sqrt(a**2 + b**2) if (a**2 + b**2) > 0 else 0
        
        # Calculate arch type based on the middle point's position
        # If middle point is above the line, it's arched; if below, it's flat or downward
        arch_direction = np.sign(a * p2[0] + b * p2[1] + c)
        
        if arch_direction > 0:
            arch_type = "Downward"
        elif arch_direction < 0:
            arch_type = "Arched"
        else:
            arch_type = "Straight"
        
        # Calculate thickness (average distance between top and bottom contours)
        thickness = h
        
        # Calculate curvature at different points
        curvatures = []
        for i in range(1, len(sorted_points) - 1):
            p_prev = sorted_points[i - 1]
            p_curr = sorted_points[i]
            p_next = sorted_points[i + 1]
            
            # Calculate vectors
            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Avoid division by zero
            if mag1 * mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                # Clamp to avoid domain errors
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle)
                curvatures.append(angle)
        
        # Calculate average curvature
        avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0
        
        # Determine eyebrow shape based on metrics
        if aspect_ratio > 4.5:
            shape = "Horizontal/Straight"
        elif aspect_ratio > 3.5:
            if arch_height > h * 0.2:
                shape = "Soft Arch"
            else:
                shape = "Flat"
        elif aspect_ratio > 2.5:
            if arch_height > h * 0.3:
                shape = "Medium Arch"
            else:
                shape = "Rounded"
        else:
            if arch_height > h * 0.4:
                shape = "High Arch"
            else:
                shape = "S-Shaped"
        
        # Return shape analysis results
        return {
            'shape': shape,
            'aspect_ratio': aspect_ratio,
            'arch_type': arch_type,
            'arch_height': arch_height,
            'thickness': thickness,
            'curvature': avg_curvature,
            'compactness': compactness,
            'convexity': convexity
        }
    
    def visualize_shape(self, image, landmarks, shape_info):
        """Visualize the shape analysis on the image"""
        if not landmarks or not shape_info:
            return image
        
        # Create a copy of the image
        vis_image = image.copy()
        
        # Convert landmarks to numpy array
        points = np.array(landmarks, dtype=np.int32)
        
        # Draw the eyebrow contour
        cv2.polylines(vis_image, [points], True, (0, 255, 255), 1)
        
        # Draw the convex hull
        hull = cv2.convexHull(points)
        cv2.polylines(vis_image, [hull], True, (255, 0, 255), 1)
        
        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Sort points by x-coordinate
        sorted_points = sorted(landmarks, key=lambda p: p[0])
        
        # Take points at 25%, 50%, and 75% positions
        p1 = sorted_points[len(sorted_points) // 4]
        p2 = sorted_points[len(sorted_points) // 2]
        p3 = sorted_points[3 * len(sorted_points) // 4]
        
        # Draw these points
        cv2.circle(vis_image, p1, 2, (0, 0, 255), -1)
        cv2.circle(vis_image, p2, 2, (0, 0, 255), -1)
        cv2.circle(vis_image, p3, 2, (0, 0, 255), -1)
        
        # Draw the line connecting first and last points
        cv2.line(vis_image, p1, p3, (255, 0, 0), 1)
        
        # Draw the arch height line
        # Line equation: ax + by + c = 0
        a = p3[1] - p1[1]
        b = p1[0] - p3[0]
        c = p3[0] * p1[1] - p1[0] * p3[1]
        
        # Calculate the point on the line that's closest to p2
        if b != 0:
            x0 = (b * (b * p2[0] - a * p2[1]) - a * c) / (a**2 + b**2)
            y0 = (a * (-b * p2[0] + a * p2[1]) - b * c) / (a**2 + b**2)
            cv2.line(vis_image, p2, (int(x0), int(y0)), (0, 255, 0), 1)
        
        # Add shape information to the image
        cv2.putText(vis_image, f"Shape: {shape_info['shape']}", (x, y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Arch: {shape_info['arch_type']}", (x, y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def get_shape_description(self, shape_info):
        """Get a textual description of the eyebrow shape"""
        if not shape_info:
            return "Unable to analyze eyebrow shape"
        
        shape = shape_info['shape']
        arch_type = shape_info['arch_type']
        thickness = shape_info['thickness']
        
        # Thickness description
        if thickness < 5:
            thickness_desc = "very thin"
        elif thickness < 10:
            thickness_desc = "thin"
        elif thickness < 15:
            thickness_desc = "medium"
        else:
            thickness_desc = "thick"
        
        # Shape descriptions
        shape_descriptions = {
            "Horizontal/Straight": "straight eyebrows that extend horizontally with minimal arch",
            "Flat": "flat eyebrows with minimal curvature",
            "Soft Arch": "gently arched eyebrows with a soft curve",
            "Medium Arch": "moderately arched eyebrows with a defined peak",
            "High Arch": "dramatically arched eyebrows with a high peak",
            "Rounded": "rounded eyebrows that follow a curved path",
            "S-Shaped": "eyebrows with an S-shaped curve"
        }
        
        description = f"{thickness_desc} {shape.lower()} eyebrows"
        if shape in shape_descriptions:
            description += f" - {shape_descriptions[shape]}"
        
        if arch_type == "Arched":
            description += ". The arch is well-defined and rises upward."
        elif arch_type == "Downward":
            description += ". The arch tends to slope downward."
        else:
            description += ". The arch is relatively straight."
        
        return description
