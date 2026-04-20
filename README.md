# YOLOv11 Streamlit Cloud Deployment

A clean, modular, and production-ready scaffold for deploying YOLOv11 object detection models on Streamlit Cloud.

## 📁 Project Structure

```
yolo-terminal-deployment/
│
├── app.py                 # CLI testing script to verify model locally
├── streamlit_app.py       # Main Streamlit web application
├── requirements.txt       # Python dependencies (Streamlit, Ultralytics, etc.)
├── packages.txt           # OS-level dependencies (libgl1 for OpenCV on cloud)
├── README.md              # Project documentation
│
├── model/                 # Contains model weights and labels
│   ├── best.pt            # Your trained YOLOv11 model
│   └── labels.txt         # Text file containing class names (one per line)
│
├── utils/                 # Core functionality modules
│   ├── detector.py        # YOLOModel wrapper class for inference
│   └── visualization.py   # Utility to draw bounding boxes and labels
│
└── assets/                # Static assets
    └── demo.png           # Demo image for testing
```

## 🚀 How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your model and image:**
   - Place your `best.pt` file inside the `model/` folder.
   - Place your `labels.txt` inside the `model/` folder.
   - Add a sample image named `demo.png` inside the `assets/` folder.

3. **Run local CLI test:**
   Verifies that the model loads and predicts correctly.
   ```bash
   python app.py
   ```

4. **Run Streamlit App:**
   Start the interactive web application.
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Streamlit Cloud Deployment Steps

1. **Commit to GitHub:**
   Push this entire repository (including `model/best.pt`, `model/labels.txt`, and `assets/demo.png`) to a GitHub repository.

2. **Deploy on Streamlit:**
   - Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
   - Click **New app**.
   - Select your GitHub repository, branch, and set `streamlit_app.py` as the Main file path.
   - Click **Deploy!**

*Note: The `packages.txt` file automatically instructs the Streamlit Cloud Linux environment to install the `libgl1` system package, which is strictly required for OpenCV to function seamlessly.*
