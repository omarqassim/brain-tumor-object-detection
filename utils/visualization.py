import cv2
import numpy as np
from PIL import Image

def draw_boxes(image, detections, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on an image.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert('RGB'))
        # Convert RGB to BGR for OpenCV
        img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    else:
        # Assuming it's already a numpy array (BGR if from cv2)
        img_cv = image.copy()
        
    # Adaptive thickness and font scaling based on image size
    height, width = img_cv.shape[:2]
    thickness = max(1, int(min(width, height) / 500))
    font_scale = max(0.4, min(width, height) / 1000.0)
    
    # Colors for different classes (pseudo-random based on class_id)
    colors = {}
    
    for det in detections:
        if det["confidence"] < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, det["box"])
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        
        # Generate color for this class if not seen before
        if cls_id not in colors:
            np.random.seed(cls_id)
            colors[cls_id] = tuple(map(int, np.random.randint(0, 255, 3)))
            
        color = colors[cls_id]
        
        # Draw box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f"{cls_name} {conf:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw text background
        cv2.rectangle(
            img_cv, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(
            img_cv, 
            label, 
            (x1, y1 - baseline - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            max(1, thickness - 1)
        )
        
    # If the input was PIL, return RGB numpy array, else return as is
    if isinstance(image, Image.Image):
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_cv
