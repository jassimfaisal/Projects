
try:
    import cv2
    st.write("cv2 loaded successfully")
except Exception as e:
    st.error(f"cv2 import failed: {e}")
    st.stop()
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="AI-Based General Movement Analysis 📈" ,
    layout="wide",
    initial_sidebar_state="expanded"
)

POSE_MODEL_PATH = "yolo26s-pose.pt"
CLASSIFIER_PATH = "posture_model.pkl"


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


FRAME_SKIP_VIDEO = 10
DISPLAY_SKIP_VIDEO = 10
RESIZE_WIDTH_VIDEO = 640
MAX_FRAMES_VIDEO = 150

FRAME_SKIP_CAMERA = 3
RESIZE_WIDTH_CAMERA = 640


@st.cache_resource
def load_pose_model():
    return YOLO(POSE_MODEL_PATH)

@st.cache_resource
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def get_point(xy, idx):
    if idx >= len(xy):
        return None
    return (float(xy[idx][0]), float(xy[idx][1]))


def resize_frame(frame, target_width=640):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    target_height = int(h * scale)
    return cv2.resize(frame, (target_width, target_height))


def extract_features_from_result(result):
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    xy = result.keypoints.xy[0].cpu().numpy()

    left_shoulder = get_point(xy, LEFT_SHOULDER)
    right_shoulder = get_point(xy, RIGHT_SHOULDER)
    left_hip = get_point(xy, LEFT_HIP)
    right_hip = get_point(xy, RIGHT_HIP)
    left_knee = get_point(xy, LEFT_KNEE)
    right_knee = get_point(xy, RIGHT_KNEE)
    left_ankle = get_point(xy, LEFT_ANKLE)
    right_ankle = get_point(xy, RIGHT_ANKLE)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    knee_angles = [v for v in [left_knee_angle, right_knee_angle] if v is not None]
    avg_knee_angle = float(np.mean(knee_angles)) if knee_angles else None

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    knee_center = midpoint(left_knee, right_knee)
    torso_angle = calculate_angle(shoulder_center, hip_center, knee_center)

    if avg_knee_angle is None or torso_angle is None:
        return None

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "avg_knee_angle": avg_knee_angle,
        "torso_angle": torso_angle
    }

def predict_posture(classifier_model, feature_dict):
    X = pd.DataFrame([feature_dict])
    prediction = classifier_model.predict(X)[0]

    torso_angle = feature_dict["torso_angle"]

    score = int((torso_angle / 180) * 100)

    if score > 100:
        score = 100
    if score < 0:
        score = 0

    if score >= 80:
        prediction = "upright"
        color = (0, 255, 0)
        message = "Good posture"
    elif score >= 60:
        prediction = "slight leaning"
        color = (0, 165, 255)
        message = "Adjust posture slightly"
    else:
        prediction = "leaning"
        color = (0, 0, 255)
        message = "Leaning detected"

    return prediction, score, message, color


def show_top_header():
    st.markdown(
        """
        <div style="padding: 1rem 0 0.5rem 0;">
            <h1 style="margin-bottom:0.25rem;">AI-Based General Movement Analysis</h1>
            <p style="font-size:1.05rem; color:#666;">
Real-time movement and posture analysis using AI.            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_project_info():
    with st.expander("Project Overview", expanded=False):
        st.write("""
       This system uses AI to track body movement in real time and analyze posture. It processes video or live camera input and gives instant feedback on how well the movement is performed.
        """)


def render_status_box(prediction, score, message):
    if prediction == "upright":
        st.success(f"**Posture:** {prediction} | **Score:** {score} | **Feedback:** {message}")
    else:
        st.error(f"**Posture:** {prediction} | **Score:** {score} | **Feedback:** {message}")


def show_summary(df, title_prefix="Session"):
    st.subheader(f"{title_prefix} Summary")

    col1, col2, col3 = st.columns(3)

    avg_score = round(df["score"].mean(), 2)
    upright_pct = round((df["prediction"] == "upright").mean() * 100, 2)
    total_frames = len(df)

    col1.metric("Average Score", avg_score)
    col2.metric("Upright %", upright_pct)
    col3.metric("Frames Analyzed", total_frames)

    st.write("Prediction counts:")
    st.write(df["prediction"].value_counts())

    chart_df = df.reset_index().rename(columns={"index": "frame"})
    chart_df = chart_df[["frame", "score"]].set_index("frame")

    st.subheader("Posture Score Over Time")
    st.line_chart(chart_df)

    st.subheader(f"{title_prefix} Results")
    st.dataframe(df.tail(20), use_container_width=True)


def run_uploaded_video(pose_model, classifier):
    st.subheader("Upload Video Analysis")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is None:
        st.info("Upload a video file to start the analysis.")
        return

    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return

    preview_col, info_col = st.columns([2, 1])
    with preview_col:
        frame_placeholder = st.empty()
    with info_col:
        status_placeholder = st.empty()
        live_result_placeholder = st.empty()

    results_rows = []
    frame_idx = 0
    processed_count = 0
    last_prediction = "N/A"
    last_score = 0
    last_message = "Waiting for valid pose..."

    progress_bar = st.progress(0)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_SKIP_VIDEO != 0:
            frame_idx += 1
            continue

        if processed_count >= MAX_FRAMES_VIDEO:
            break

        frame = resize_frame(frame, RESIZE_WIDTH_VIDEO)
        results = pose_model.predict(frame, conf=0.4, verbose=False)

        annotated = frame.copy()

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            result = results[0]
            annotated = result.plot()

            feature_dict = extract_features_from_result(result)

            if feature_dict is not None:
                prediction, score, message, color = predict_posture(classifier, feature_dict)
                last_prediction = prediction
                last_score = score
                last_message = message

                cv2.putText(annotated, f"Posture: {prediction}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated, f"Score: {score}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated, message, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                results_rows.append({
                    **feature_dict,
                    "prediction": prediction,
                    "score": score
                })

        if processed_count % DISPLAY_SKIP_VIDEO == 0:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            status_placeholder.info(
                f"Processed frames: {processed_count + 1} / {MAX_FRAMES_VIDEO}"
            )
            with live_result_placeholder.container():
                render_status_box(last_prediction, last_score, last_message)

        processed_count += 1
        frame_idx += 1
        progress_bar.progress(min(processed_count / MAX_FRAMES_VIDEO, 1.0))

    cap.release()
    progress_bar.empty()

    if results_rows:
        df = pd.DataFrame(results_rows)
        df.to_csv("session_results.csv", index=False)

        show_summary(df, "Uploaded Video")

        st.download_button(
            "Download video results as CSV",
            data=df.to_csv(index=False),
            file_name="session_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("No valid posture data was extracted from the uploaded video.")


def run_live_camera(pose_model, classifier):
    st.subheader("Live Camera Analysis")
    st.info("Press **Start Live Camera**. A webcam window will open. Press **Q** to stop and save the session results.")

    if st.button("Start Live Camera", use_container_width=True):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam.")
            return

        frame_idx = 0
        results_rows = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % FRAME_SKIP_CAMERA != 0:
                frame_idx += 1
                continue

            frame = resize_frame(frame, RESIZE_WIDTH_CAMERA)
            results = pose_model.predict(frame, conf=0.4, verbose=False)

            annotated = frame.copy()

            if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                result = results[0]
                annotated = result.plot()

                feature_dict = extract_features_from_result(result)

                if feature_dict is not None:
                    prediction, score, message, color = predict_posture(classifier, feature_dict)

                    cv2.putText(annotated, f"Posture: {prediction}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(annotated, f"Score: {score}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(annotated, message, (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    results_rows.append({
                        **feature_dict,
                        "prediction": prediction,
                        "score": score
                    })

            cv2.imshow("Live Camera Movement Analysis", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        if results_rows:
            df = pd.DataFrame(results_rows)
            df.to_csv("live_camera_results.csv", index=False)

            show_summary(df, "Live Camera")

            st.download_button(
                "Download live camera results as CSV",
                data=df.to_csv(index=False),
                file_name="live_camera_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid posture data was captured from the live camera session.")



show_top_header()
show_project_info()

if not os.path.exists(CLASSIFIER_PATH):
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

pose_model = load_pose_model()
classifier = load_classifier()

with st.sidebar:
    st.header("Control Panel")
    mode = st.selectbox("Choose mode", ["Upload Video", "Live Camera "])
    st.markdown("---")
   

if mode == "Upload Video":
    run_uploaded_video(pose_model, classifier)
else:
    run_live_camera(pose_model, classifier)
