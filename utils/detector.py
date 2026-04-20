import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load labels
        self.labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                for idx, line in enumerate(f):
                    self.labels[idx] = line.strip()
        else:
            print(f"Warning: Labels file not found at {labels_path}")
            
        # Load model
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model.to(self.device)
        else:
            print(f"Warning: Model file not found at {model_path}")
            self.model = None

    def predict(self, image):
        if self.model is None:
            return []

        # Convert PIL Image to RGB if needed
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            
        # Run inference
        results = self.model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates, confidence, and class id
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get class name, fallback to class_id string if not found
                cls_name = self.labels.get(cls_id, str(cls_id))
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })
                
        return detections
