import os
import streamlit as st
from PIL import Image
import numpy as np

# Import internal modules (relative paths from root)
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

# --- Page Config ---
st.set_page_config(
    page_title="YOLOv11 Object Detection",
    page_icon="🔍",
    layout="centered"
)

# --- CSS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🔍 YOLOv11 Object Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image to run object detection or use the demo image.")

# --- Initialization ---
@st.cache_resource
def load_model():
    # Caching the model to avoid reloading on every interaction
    return YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")

model = load_model()

# --- Sidebar ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# --- Main UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
use_demo = st.checkbox("Use demo image instead")

image = None

if uploaded_file is not None and not use_demo:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif use_demo:
    demo_path = "assets/demo.png"
    if os.path.exists(demo_path):
        try:
            image = Image.open(demo_path)
        except Exception as e:
            st.error(f"Error loading demo image: {e}")
    else:
        st.warning("Demo image not found at assets/demo.png. Please upload an image.")

if image is not None:
    # Create two columns for Before/After
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    run_detection = st.button("🚀 Run Detection", use_container_width=True)
    
    if run_detection:
        with st.spinner("Running inference..."):
            if model.model is None:
                st.error("Model not found! Please ensure 'model/best.pt' is in the project directory.")
            else:
                try:
                    # Run prediction
                    detections = model.predict(image)
                    
                    if not detections:
                        st.info("No objects detected.")
                    else:
                        # Draw boxes
                        result_img = draw_boxes(image, detections, conf_threshold=conf_threshold)
                        
                        with col2:
                            st.subheader("Annotated Image")
                            st.image(result_img, use_container_width=True)
                            
                        # Show raw detections inside an expander
                        with st.expander("Show Raw Detections JSON"):
                            filtered_dets = [d for d in detections if d["confidence"] >= conf_threshold]
                            st.json(filtered_dets)
                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
