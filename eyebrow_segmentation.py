import cv2
import numpy as np
from sklearn.cluster import KMeans

class EyebrowSegmentation:
    def __init__(self):
        pass
    
    def create_eyebrow_mask(self, image, eyebrow_landmarks, padding=5):
        """Create a mask for the eyebrow region based on landmarks"""
        if not eyebrow_landmarks or len(eyebrow_landmarks) < 5:
            return None, None
        
        # Create an empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Convert landmarks to numpy array
        points = np.array(eyebrow_landmarks, dtype=np.int32)
        
        # Get bounding rectangle of eyebrow points
        x, y, w, h = cv2.boundingRect(points)
        
        # Add padding to the bounding rectangle
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Create a convex hull around the eyebrow points
        hull = cv2.convexHull(points)
        
        # Fill the convex hull on the mask
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        
        # Ensure the mask is not empty
        if cv2.countNonZero(mask) == 0: # type: ignore
            return None, None
            
        return mask, (x, y, w, h)
    
    def apply_mask(self, image, mask):
        """Apply mask to the image"""
        if mask is None:
            return None
        
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
    
    def extract_eyebrow_region(self, image, mask, bbox):
        """Extract the eyebrow region from the image using mask and bounding box"""
        if mask is None or bbox is None:
            return None, None
        
        x, y, w, h = bbox
        
        # Ensure bbox is within image bounds
        if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
            x = max(0, x)
            y = max(0, y)
            w = min(image.shape[1] - x, w)
            h = min(image.shape[0] - y, h)
        
        # Extract the region of interest
        roi = image[y:y+h, x:x+w].copy()
        roi_mask = mask[y:y+h, x:x+w].copy()
        
        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
        
        return masked_roi, roi_mask
    
    def refine_eyebrow_mask(self, image, mask):
        """Refine the eyebrow mask using color thresholding"""
        if mask is None or cv2.countNonZero(mask) == 0:
            return None
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply the initial mask
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # Extract only the masked region
        masked_pixels = masked_hsv[mask > 0]
        
        if len(masked_pixels) == 0:
            return mask
        
        # Reshape for K-means
        pixels = masked_pixels.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        k = 2  # We want to separate eyebrow from skin
        if len(pixels) > k:
            # Use scikit-learn's KMeans instead of OpenCV's kmeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Find the darker cluster (likely eyebrow)
            darker_cluster = np.argmin([center[2] for center in centers])  # V channel in HSV
            
            # Create a refined mask based on the darker cluster
            refined_mask = np.zeros_like(mask)
            flat_mask = mask.flatten()
            flat_refined = refined_mask.flatten()
            
            # Set pixels belonging to the darker cluster
            mask_indices = np.where(flat_mask > 0)[0]
            
            # Ensure we have valid indices
            if len(mask_indices) > 0 and len(labels) > 0:
                darker_indices = mask_indices[labels.flatten() == darker_cluster]
                flat_refined[darker_indices] = 255
                
                refined_mask = flat_refined.reshape(mask.shape)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Ensure the refined mask is not empty
                if cv2.countNonZero(refined_mask) == 0:
                    return mask
                    
                return refined_mask
        
        return mask
    
    def alpha_matting(self, image, mask, bbox=None):
        """Create an alpha matte for the eyebrow"""
        if mask is None or cv2.countNonZero(mask) == 0:
            return None, None
        
        # Create a trimap: 0 for background, 255 for foreground, 128 for unknown
        trimap = np.zeros_like(mask)
        
        # Erode mask for definite foreground
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.erode(mask, kernel, iterations=2)
        
        # Dilate mask for possible foreground/background
        unknown = cv2.dilate(mask, kernel, iterations=2)
        
        # Set values in trimap
        trimap[fg > 0] = 255
        trimap[unknown > 0] = 128
        
        # Since we don't have a true alpha matting algorithm in OpenCV,
        # we'll simulate it using the mask and edge detection
        edges = cv2.Canny(mask, 100, 200)
        
        # Create alpha matte
        alpha = np.zeros_like(mask, dtype=np.float32)
        alpha[mask > 0] = 1.0
        
        # Smooth the edges
        blur_edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
        
        # Blend the edges
        alpha = alpha - (blur_edges * 0.5)
        alpha = np.clip(alpha, 0, 1)
        
        # Convert to 8-bit
        alpha_8bit = (alpha * 255).astype(np.uint8)
        
        # Apply alpha to image
        b, g, r = cv2.split(image)
        rgba = cv2.merge((r, g, b, alpha_8bit))
        
        # If bbox is provided, crop the alpha matte to the bounding box
        if bbox is not None:
            x, y, w, h = bbox
            
            # Ensure bbox is within image bounds
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                x = max(0, x)
                y = max(0, y)
                w = min(image.shape[1] - x, w)
                h = min(image.shape[0] - y, h)
                
            if w > 0 and h > 0:  # Ensure valid dimensions
                cropped_rgba = rgba[y:y+h, x:x+w].copy()
                cropped_alpha = alpha_8bit[y:y+h, x:x+w].copy()
                return cropped_rgba, cropped_alpha
        
        return rgba, alpha_8bit
    
    def get_cropped_mask(self, mask, bbox):
        """Get a cropped version of the mask using the bounding box"""
        if mask is None or bbox is None:
            return None
            
        x, y, w, h = bbox
        
        # Ensure bbox is within mask bounds
        if x < 0 or y < 0 or x+w > mask.shape[1] or y+h > mask.shape[0]:
            x = max(0, x)
            y = max(0, y)
            w = min(mask.shape[1] - x, w)
            h = min(mask.shape[0] - y, h)
            
        if w <= 0 or h <= 0:  # Invalid dimensions
            return None
            
        return mask[y:y+h, x:x+w].copy()
