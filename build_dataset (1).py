from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


MODEL_PATH = "yolo26n-pose.pt"
VIDEO_FOLDER = "videos"
OUTPUT_CSV = "movement_dataset.csv"
FRAME_SKIP = 5

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
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)


def get_point(xy, conf, idx):
    if idx >= len(xy):
        return None
    return (float(xy[idx][0]), float(xy[idx][1]))


def main():
    model = YOLO(MODEL_PATH)
    video_paths = list(Path(VIDEO_FOLDER).glob("*.mp4"))

    rows = []

    for video_path in video_paths:
        print("Processing:", video_path)
        cap = cv2.VideoCapture(str(video_path))

        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            results = model.predict(frame, conf=0.4, verbose=False)

            if results and results[0].keypoints is not None:
                xy = results[0].keypoints.xy[0].cpu().numpy()

                left_hip = get_point(xy, None, LEFT_HIP)
                right_hip = get_point(xy, None, RIGHT_HIP)
                left_knee = get_point(xy, None, LEFT_KNEE)
                right_knee = get_point(xy, None, RIGHT_KNEE)
                left_ankle = get_point(xy, None, LEFT_ANKLE)
                right_ankle = get_point(xy, None, RIGHT_ANKLE)

                left_shoulder = get_point(xy, None, LEFT_SHOULDER)
                right_shoulder = get_point(xy, None, RIGHT_SHOULDER)

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                avg_knee = np.mean([v for v in [left_knee_angle, right_knee_angle] if v is not None]) if left_knee_angle and right_knee_angle else None

                hip_center = midpoint(left_hip, right_hip)
                shoulder_center = midpoint(left_shoulder, right_shoulder)
                knee_center = midpoint(left_knee, right_knee)

                torso_angle = calculate_angle(shoulder_center, hip_center, knee_center)

                label = "leaning" if torso_angle and torso_angle < 145 else "upright"

                rows.append({
                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,
                    "avg_knee_angle": avg_knee,
                    "torso_angle": torso_angle,
                    "label": label
                })

            frame_idx += 1

        cap.release()

    df = pd.DataFrame(rows)
    df = df.dropna()
    df.to_csv(OUTPUT_CSV, index=False)

    print("Dataset saved:", OUTPUT_CSV)
    print(df.head())


if __name__ == "__main__":
    main()