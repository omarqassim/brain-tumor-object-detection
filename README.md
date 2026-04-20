# 🧠 YOLOv11 Brain Tumor Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=flat-square&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Headless-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**An end-to-end computer vision pipeline for automated brain tumor detection in MRI scans,
powered by YOLOv11 and deployed as an interactive web application on Streamlit Cloud.**

🔗 **Live Demo:** [Add your Streamlit Cloud link here]

</div>

---

## 📌 Overview

This project fine-tunes **YOLOv11** on a labeled MRI brain tumor dataset to perform real-time
object detection across four tumor classes. The trained model is packaged into a clean,
production-ready **Streamlit** web application that allows users to upload MRI scans and
receive instant annotated predictions directly in the browser — no setup required.

---

## 🎯 Detected Classes

| Class | Description |
|-------|-------------|
| `glioma` | Malignant tumor originating in glial cells |
| `meningioma` | Tumor arising from the meninges |
| `pituitary` | Tumor located in the pituitary gland |
| `notumor` | Healthy MRI scan with no tumor present |

---

## 🏗️ Project Structure

```
yolov11-brain-tumor-detector/
│
├── app.py                  # CLI script for local inference testing
├── streamlit_app.py        # Main Streamlit web application
├── requirements.txt        # Python dependencies
├── packages.txt            # OS-level dependencies (libgl1 for OpenCV)
├── README.md               # Project documentation
│
├── model/
│   ├── best.pt             # Trained YOLOv11 weights
│   └── labels.txt          # Class names (one per line)
│
├── utils/
│   ├── detector.py         # YOLOModel class — model loading & inference
│   └── visualization.py    # draw_boxes() — bounding box rendering
│
└── assets/
    └── demo.png            # Sample MRI image for demo mode
```

---

## ⚙️ Tech Stack

- **Model:** YOLOv11s (Ultralytics) — fine-tuned on MRI brain tumor dataset
- **Dataset:** Labeled MRI Brain Tumor Dataset via Roboflow
- **Framework:** Streamlit
- **Computer Vision:** OpenCV (headless), Pillow
- **Training Environment:** Google Colab (GPU)
- **Deployment:** Streamlit Community Cloud

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/omarqassim/yolov11-brain-tumor-detector.git
cd yolov11-brain-tumor-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Test inference via CLI

```bash
python app.py
```

Runs inference on `assets/demo.png` and saves the annotated result to `output.jpg`.

### 4. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub (including `model/best.pt`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Select your repository and set **Main file path** to `streamlit_app.py`
5. Click **Deploy**

> The `packages.txt` file automatically installs `libgl1` on the Linux cloud environment,
> which is required for OpenCV to function correctly.

---

## 🖥️ App Features

- 📤 Upload any MRI image (JPG / PNG)
- 🖼️ Demo mode using a pre-loaded sample image
- 🎚️ Adjustable confidence threshold via sidebar slider
- ⚡ Real-time inference with loading spinner
- 📊 Detection results table (class, confidence, bounding box coordinates)
- 🔍 Side-by-side view: original vs. annotated image

---

## 📊 Model Training Summary

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv11s |
| Epochs | 30 |
| Image Size | 640×640 |
| Batch Size | 16 |
| Optimizer | SGD |
| Augmentation | Mosaic, Flip, Rotation, HSV |
| Early Stopping | Patience = 10 |

---

## 📁 Dataset

Dataset sourced from **Roboflow Universe**:
[Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/omarqassim/labeled-mri-brain-tumor-dataset-hie10)

---

## 👤 Author

**Omar Qassim**
Computer Science Engineering Student | AI & Data Science | Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-omarqassim-black?style=flat-square&logo=github)](https://github.com/omarqassim)

---

## 📄 License

This project is licensed under the **MIT License**.
