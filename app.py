import os
import cv2
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    print("--- YOLOv11 Local Testing Script ---")
    
    demo_image_path = "assets/demo.png"
    output_image_path = "output.jpg"
    
    if not os.path.exists(demo_image_path):
        print(f"Error: Demo image not found at {demo_image_path}")
        return

    print("Loading model...")
    detector = YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")
    
    if detector.model is None:
        print("Failed to load model. Exiting.")
        return

    print("Loading demo image...")
    try:
        image = Image.open(demo_image_path)
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    print("Running inference...")
    detections = detector.predict(image)
    
    print("\n--- Detections ---")
    if not detections:
        print("No objects detected.")
    else:
        for det in detections:
            print(f"Class: {det['class_name']} (ID: {det['class_id']}), "
                  f"Confidence: {det['confidence']:.2f}, "
                  f"Box: {[int(x) for x in det['box']]}")
            
    print("\nDrawing boxes...")
    # draw_boxes returns an RGB numpy array since we passed a PIL Image
    annotated_img_rgb = draw_boxes(image, detections, conf_threshold=0.25)
    
    # Convert RGB to BGR to save with OpenCV
    annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
    
    print(f"Saving output to {output_image_path}...")
    cv2.imwrite(output_image_path, annotated_img_bgr)
    
    print("Done! Check output.jpg")

if __name__ == "__main__":
    main()
