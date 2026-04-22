# 🧠 AI-Based General Movement Analysis

> AI-powered system that analyzes human movement and provides real-time posture feedback from video or live camera.

---

## 📌 Problem Statement & Motivation

Poor posture during movement can lead to injuries and reduced performance, especially in activities like training or daily physical tasks. 

The motivation behind this project is to create a simple and accessible AI system that can automatically analyze movement and provide instant posture feedback, helping users improve their performance and reduce the risk of injury.

---

## 📊 Dataset Description & Sources

The dataset was created using real movement videos, primarily from fitness and CrossFit activities.

Using a pose estimation model, body keypoints were extracted from each frame, and important features such as knee angles and torso angle were computed. These features were then used to label the data into two classes: `upright` and `leaning`.

This approach allowed the creation of a custom dataset tailored specifically for posture analysis.

---

## 🤖 Model Architecture & Training Pipeline

The system combines pose estimation and machine learning to analyze movement and classify posture.

First, a pretrained pose model is used to extract body keypoints from each frame. These keypoints are then converted into meaningful features, such as joint angles. Finally, a Random Forest classifier is trained on this data to predict posture as `upright` or `leaning`.

This pipeline allows the system to process video input and generate real-time posture feedback.

---

## 📈 Evaluation Results & Baseline Comparison

The model achieved very high accuracy on the test dataset, indicating that it can reliably distinguish between `upright` and `leaning` posture.

In addition to test results, the system was evaluated on real video inputs, where it consistently produced stable and meaningful predictions across multiple frames.

As a baseline, a simple rule-based approach using angle thresholds can be used for posture detection. However, the trained model provides more flexibility and robustness by learning patterns directly from the data rather than relying on fixed rules.

---

## 🚀 How to Run the Project

1. Open your project folder:
2. cd your_project_folder


2. Install dependencies:
3. pip install ultralytics opencv-python numpy pandas scikit-learn streamlit joblib matplotlib.


3. Train the model:
python train_model.py

---


4. Run the app:
5. streamlit run app.py


6. Use the system:
- Upload a video  
- Or use live camera

  ---

## 📁 Repo Structure

- `app.py` → main application  
- `train_model.py` → model training  
- `build_dataset.py` → dataset creation  
- `posture_model.pkl` → trained model  
- `movement_dataset.csv` → dataset  
- `videos/` → input data

  ---

  ## ⚠️ Limitations & Future Work

### Limitations
- The system currently classifies only two posture types: `upright` and `leaning`  
- Performance depends on camera angle, lighting, and visibility  
- The scoring system is simple and does not capture full movement quality  

### Future Work
- Add more posture and movement categories  
- Improve the scoring system with more detailed feedback  
- Use advanced models for deeper movement analysis  
- Enhance robustness for different environments and users

  ---

  ## 🎥 Demo & Screenshots

### 📸 Application Preview

![App Screenshot](Images/app_sec.png)

### 🎬 Demo

![Demo GIF](images/vid.mp4)

---

The system supports:
- Video upload analysis  
- Live camera posture detection  

  ---

  ## 📜 License & Acknowledgments

### License
This project is developed for educational purposes.

### Acknowledgments
- Ultralytics YOLO for pose estimation  
- OpenCV for video processing  
- Scikit-learn for machine learning  
- Streamlit for building the application  
