import streamlit as st
import cv2
import joblib
import pandas as pd
import numpy as np
from ultralytics import YOLO

st.title("AI-Based General Movement Analysis")

pose_model = YOLO("yolo26n-pose.pt")
classifier = joblib.load("posture_model.pkl")

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def get_point(xy, idx):
    if idx >= len(xy):
        return None
    return (float(xy[idx][0]), float(xy[idx][1]))


uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    frame_placeholder = st.empty()
    results_table = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = pose_model.predict(frame, conf=0.4, verbose=False)

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            result = results[0]
            annotated = result.plot()

            xy = result.keypoints.xy[0].cpu().numpy()

            left_hip = get_point(xy, LEFT_HIP)
            right_hip = get_point(xy, RIGHT_HIP)
            left_knee = get_point(xy, LEFT_KNEE)
            right_knee = get_point(xy, RIGHT_KNEE)
            left_ankle = get_point(xy, LEFT_ANKLE)
            right_ankle = get_point(xy, RIGHT_ANKLE)
            left_shoulder = get_point(xy, LEFT_SHOULDER)
            right_shoulder = get_point(xy, RIGHT_SHOULDER)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            if left_knee_angle is not None and right_knee_angle is not None:
                avg_knee_angle = np.mean([left_knee_angle, right_knee_angle])
            else:
                avg_knee_angle = None

            hip_center = midpoint(left_hip, right_hip)
            shoulder_center = midpoint(left_shoulder, right_shoulder)
            knee_center = midpoint(left_knee, right_knee)

            torso_angle = calculate_angle(shoulder_center, hip_center, knee_center)

            if None not in [left_knee_angle, right_knee_angle, avg_knee_angle, torso_angle]:
                X = pd.DataFrame([{
                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,
                    "avg_knee_angle": avg_knee_angle,
                    "torso_angle": torso_angle
                }])

                prediction = classifier.predict(X)[0]

                color = (0, 255, 0) if prediction == "upright" else (0, 0, 255)

                cv2.putText(
                    annotated,
                    f"Posture: {prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                results_table.append({
                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,
                    "avg_knee_angle": avg_knee_angle,
                    "torso_angle": torso_angle,
                    "prediction": prediction
                })

            frame_placeholder.image(annotated, channels="BGR")

    cap.release()

    if results_table:
        st.subheader("Results")
        df = pd.DataFrame(results_table)
        st.dataframe(df.tail(20), use_container_width=True)